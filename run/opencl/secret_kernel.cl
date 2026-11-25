/*
 * OpenCL kernel for LinkeSOFT "Secret!" password manager
 * (Android/iPhone/BlackBerry/Desktop AES family), companion to
 * secret_fmt_plug.c / opencl_secret_fmt_plug.c.
 *
 * This software is Copyright (c) 2026 by Durval Menezes, and it is hereby
 * released to the general public under the terms of the WTF Program License
 * https://www.wtfpl.net/; There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * Algorithm reconstructed from com.linkesoft.secret.data.Crypt (Secret! for
 * Android 2.3.2, decompiled with androguard, 2026-11):
 *
 *   upper   = password.toUpperCase() in Latin-1 (ISO8859-1) bytes
 *   key     = MD5(upper)                          -> AES-128 key, no salt, no KDF
 *   verifier("encrpasswd") = AES-128-CBC-encrypt( key, IV = 16 zero bytes,
 *                                   plaintext = upper + 0x00, PKCS#5 padding )
 *
 * AES backend: AES_BITSLICE (see opencl_aes.h / opencl_aes_bitslice.h), not
 * the default local-table backend. Every work-item calls
 * AES_set_decrypt_key() at least once per kernel launch, and the default
 * backend pulls in three barrier() pairs plus collaborative local-memory
 * table reloads per call - fixed overhead that dwarfs the one or two
 * 16-byte block decrypts this format actually does. Bitslice has no local
 * memory, no barriers, no table setup: pure register/ALU work. This is the
 * same backend the stock o5logon-opencl format uses, which is also this
 * kernel's structural template for GPU-side mask acceleration below.
 *
 * GPU-side mask acceleration (mask_int_cand / FMT_MASK): the host no longer
 * hands this kernel one fully-expanded password per work-item. Up to
 * MASK_FMT_INT_PLHDR (4) of the mask's placeholder positions are expanded
 * ON THE DEVICE: each work-item receives one "base" candidate (with those
 * positions left as whatever set_key() wrote, and GPU_LOC_n saying where
 * they are) and an int_keys[] table of every combination of characters for
 * those positions, then loops NUM_INT_KEYS times internally, substituting
 * one combination per iteration before hashing. This is what actually fixes
 * --mask throughput: john's CPU-side candidate generator and set_key() are
 * only invoked once per *outer* combination (the positions NOT offloaded to
 * the GPU), instead of once per candidate overall, so the host-CPU
 * bottleneck documented in the README (mask mode was ~26-36M p/s,
 * GPU 2/3 idle) shrinks by roughly the GPU-side combination count. Without
 * --mask, mask_int_cand.num_int_cand is 1 and NUM_INT_KEYS collapses the
 * loop below to a single iteration with the extra password[]-array copy as
 * the only overhead versus the prior non-mask kernel.
 *
 * This structure (int_key_loc/int_keys arguments, the GPU_LOC_n dispatch,
 * the atomic-counter + out_index crack-reporting convention in place of a
 * per-gid flags array) is adapted line-for-line from o5logon_kernel.cl,
 * john's only other GPU-mask format, since it is the one verified reference
 * for this mechanism. One deviation: o5logon_kernel.cl's own LOC_1/LOC_2/
 * LOC_3 extraction has an operator-precedence bug -
 * "int_keys[idx] & 0xff >> 8" parses as "int_keys[idx] & (0xff >> 8)" i.e.
 * "& 0", always zero, because "&" binds looser than ">>" in C/OpenCL C - so
 * any mask using more than one GPU-side placeholder silently mangles
 * positions 1-3 upstream. Fixed here as "(int_keys[idx] >> 8) & 0xff" etc.
 * This format's own masks (repeated identical placeholder ranges, as in
 * ?1?1?1?1...) are exactly the case that would hit more than one GPU-side
 * position, so getting this right matters for real use, not just in theory.
 *
 * The salt is rejected unless blob_len is 16 or 32, so a zeroed salt buffer
 * (before the first set_salt(), e.g. during LWS benchmarking) can never
 * reach the cipher. blob_len is uniform across all work-items and checked
 * once per work-item (not per internal candidate), since it is a salt-level
 * property.
 */

#include "opencl_device_info.h"
#include "opencl_misc.h"
#include "opencl_md5.h"
#include "opencl_mask.h"
#define AES_BITSLICE
#include "opencl_aes.h"

#ifndef PLAINTEXT_LENGTH
#define PLAINTEXT_LENGTH        24
#endif

/* Must match the host struct in opencl_secret_fmt_plug.c byte for byte. */
typedef struct {
	uint   blob_len;    /* 16 or 32 */
	uchar  blob[32];    /* encrpasswd verifier, hex-decoded on host */
} secret_salt;

__kernel void secret(__global const uchar *key_buf,
                     __global const uint *const key_idx,
                     __constant secret_salt *salt,
                     __global volatile uint *crack_count_ret,
                     __global uint *const out_index,
                     __global const uint *const int_key_loc,
                     __global const uint *const int_keys)
{
	const uint gid = get_global_id(0);

	/* Reject anything but the two valid verifier sizes. Salt-level, so
	 * checked once per work-item rather than once per internal candidate. */
	if (salt->blob_len != 16 && salt->blob_len != 32)
		return;

#if NUM_INT_KEYS > 1 && !IS_STATIC_GPU_MASK
	const uint ikl = int_key_loc[gid];
	const uint loc0 = ikl & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
	const uint loc1 = (ikl & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
	const uint loc2 = (ikl & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
	const uint loc3 = (ikl & 0xff000000) >> 24;
#endif
#endif
#endif

#if !IS_STATIC_GPU_MASK
#define GPU_LOC_0 loc0
#define GPU_LOC_1 loc1
#define GPU_LOC_2 loc2
#define GPU_LOC_3 loc3
#else
#define GPU_LOC_0 LOC_0
#define GPU_LOC_1 LOC_1
#define GPU_LOC_2 LOC_2
#define GPU_LOC_3 LOC_3
#endif

	const uint base = key_idx[gid];
	const uint len = key_idx[gid + 1] - base;
	key_buf += base;

	/* Blob to private memory once per work-item (salt-level, same for
	 * every internal candidate). */
	uchar blob[32];
	for (uint i = 0; i < salt->blob_len; i++)
		blob[i] = salt->blob[i];

	for (uint idx = 0; idx < NUM_INT_KEYS; idx++) {
		const uint gidx = gid * NUM_INT_KEYS + idx;
		uchar password[PLAINTEXT_LENGTH] = { 0 };
		uint i;

		for (i = 0; i < len; i++)
			password[i] = key_buf[i];

#if NUM_INT_KEYS > 1
		password[GPU_LOC_0] = (int_keys[idx] & 0xff);
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
		password[GPU_LOC_1] = (int_keys[idx] >> 8) & 0xff;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
		password[GPU_LOC_2] = (int_keys[idx] >> 16) & 0xff;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
		password[GPU_LOC_3] = (int_keys[idx] >> 24) & 0xff;
#endif
#endif
#endif

		/*
		 * One pass: Latin-1 uppercase + pack into six little-endian
		 * words. pw is zero-initialized, so byte len (the
		 * addTrailingZero 0x00 position) is already zero whenever
		 * len < 24.
		 */
		uint pw[6];
		for (i = 0; i < 6; i++)
			pw[i] = 0;
		for (i = 0; i < len; i++) {
			uchar c = password[i];
			if ((c >= 'a' && c <= 'z') ||
			    (c >= 0xE0 && c <= 0xFE && c != 0xF7))
				c -= 0x20;
			pw[i >> 2] |= ((uint)c) << ((i & 3) << 3);
		}

		/*
		 * Single-block MD5: message (<= 24 bytes) + 0x80 + bit length
		 * all fit one 64-byte block. The full zero-fill first matters
		 * for len = 24, where the 0x80 lands in word 6 - beyond the
		 * six password words.
		 */
		uint W[16];
		for (i = 0; i < 6; i++)
			W[i] = pw[i];
		for (i = 6; i < 16; i++)
			W[i] = 0;
		W[len >> 2] |= 0x80U << ((len & 3) << 3);
		W[14] = len << 3;

		uint digest[4];
		md5_single(uint, W, digest);

		/* AES key = digest bytes, little-endian per word. */
		uchar key16[16];
		for (i = 0; i < 4; i++) {
			key16[4 * i + 0] = (uchar)digest[i];
			key16[4 * i + 1] = (uchar)(digest[i] >> 8);
			key16[4 * i + 2] = (uchar)(digest[i] >> 16);
			key16[4 * i + 3] = (uchar)(digest[i] >> 24);
		}

		/* Bitslice backend: no local-memory table cache needed. */
		AES_KEY akey;
		AES_set_decrypt_key(key16, 128, &akey);

		/* CBC with zero IV, unrolled: p0 = D(c0); p1 = D(c1) ^ c0. */
		uchar dec[32];
		AES_decrypt(blob, dec, &akey);
		if (salt->blob_len == 32) {
			AES_decrypt(blob + 16, dec + 16, &akey);
			for (i = 0; i < 16; i++)
				dec[16 + i] ^= blob[i];
		}

		/*
		 * Expected plaintext: upper(pw) + 0x00 + PKCS#5 pad to
		 * blob_len, where pad = 16 - ((len + 1) & 15) - a full 0x10
		 * block when (len + 1) is a multiple of 16.
		 */
		uint pad = 16 - ((len + 1) & 15);
		uint ok = 1;
		for (i = 0; i < salt->blob_len; i++) {
			uchar e;
			if (i < len)
				e = (uchar)(pw[i >> 2] >> ((i & 3) << 3));
			else if (i == len)
				e = 0x00;
			else
				e = (uchar)pad;
			ok &= (dec[i] == e);
		}

		if (ok)
			out_index[atomic_inc(crack_count_ret)] = gidx;
	}
}

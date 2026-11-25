/*
 * JtR cracker for LinkeSoft "Secret!" password manager
 * (Android/iPhone/BlackBerry/Desktop AES family).
 *
 * This software is Copyright (c) 2026 by Durval Menezes, and it is hereby
 * released to the general public under the terms of the WTF Program License
 * https://www.wtfpl.net/; There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * Algorithm reconstructed from com.linkesoft.secret.data.Crypt (Secret! for
 * Android 2.3.2, decompiled with androguard, 2026-11):
 *
 *   upper   = password.toUpperCase() in Latin-1 (ISO8859-1) bytes
 *   key     = MD5(upper)                          -> AES-128 key, no salt, no KDF iterations
 *   verifier("encrpasswd") = AES-128-CBC-encrypt( key, IV = 16 zero bytes,
 *                                   plaintext = upper + 0x00, PKCS#5/#7 padding )
 *
 * The app stores this verifier per record and compares it against
 * encryptPassword(candidate), so each record is independently crackable.
 *
 * Input format (one line per record; the 16- or 32-byte encrpasswd blob as hex;
 * extraction from secret.dat / secret2.pdb: see secret_dat_probe.py):
 *
 *   $secret$<32 or 64 hex chars>
 *
 * Note: the effective keyspace is uppercase-only (the app folds case), so
 * cracking "password" also reports for candidates "PASSWORD"/"Password".
 * Password length in the app is limited to 24 characters.
 */

#if AC_BUILT
#include "autoconfig.h"
#endif

#if HAVE_LIBCRYPTO || HAVE_COMMONCRYPTO
#if FMT_EXTERNS_H
extern struct fmt_main fmt_secret;
#elif FMT_REGISTERS_H
john_register_one(&fmt_secret);
#else

#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "arch.h"
#include "aes.h"
#include "md5.h"
#include "jumbo.h"
#include "params.h"
#include "common.h"
#include "formats.h"

#define FORMAT_LABEL        "Secret"
#define FORMAT_NAME         ""
#define ALGORITHM_NAME      "MD5(pw) + AES-128-CBC(zero IV)"
#define BENCHMARK_COMMENT   ""
#define BENCHMARK_LENGTH    0x107
#define PLAINTEXT_LENGTH    24
#define BINARY_SIZE         0
#define BINARY_ALIGN        1
#define SALT_SIZE           sizeof(struct custom_salt)
#define SALT_ALIGN          4
#define MIN_KEYS_PER_CRYPT  1
#define MAX_KEYS_PER_CRYPT  64

#ifndef OMP_SCALE
#define OMP_SCALE           16384
#endif

#define FORMAT_TAG          "$secret$"
#define FORMAT_TAG_LEN      (sizeof(FORMAT_TAG) - 1)

/* Minimal Latin-1 toupper (ASCII a-z + ISO-8859-1 accented range). */
static unsigned char latin1_upper[256];

static void init_tables(void)
{
	int i;
	for (i = 0; i < 256; i++)
		latin1_upper[i] = (unsigned char)i;
	for (i = 'a'; i <= 'z'; i++)
		latin1_upper[i] = i - 0x20;
	for (i = 0xE0; i <= 0xFE; i++)
		if (i != 0xF7) /* division sign has no uppercase in Latin-1 */
			latin1_upper[i] = i - 0x20;
}

struct custom_salt {
	unsigned int blob_len;
	unsigned char blob[32];
};

static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static int any_cracked, *cracked, cracked_count;
static struct custom_salt *cur_salt;

static struct fmt_tests secret_tests[] = {
	/* Test vectors generated from the reconstructed algorithm. */
	{"$secret$f7ba630de5803364d1891cfd02077fab", "cisco"},
	{"$secret$f3003d2aa438ad9306c5806b1916567c", "password"},
	{"$secret$9d83be34ca962e211cbe40f6b466869c", "secret"},
	{"$secret$24bd82c02a10825f59f307da07dcd50a", "PalmPilot1997"},
	{"$secret$51b3be590bf8eb5b9c072e7de4fc7456", "Tr0ub4dor&3"},
	{"$secret$cc9dce9431e3d9e0abd068afc56e9b0fa9b8347d7fc88fd882335aa0fd213ce8", "0123456789abcdef"},
	/* Edge cases (gen_vectors_edge.py): len=15 -> PKCS#7 pad 0x10 full
	 * block (32-byte blob); len=14 -> largest key with a 16-byte blob
	 * (pad = 1); len=24 = PLAINTEXT_LENGTH maximum. Non-ASCII keys cannot
	 * be tested here: john's self-test pipeline assumes UTF-8 keys; verify
	 * the Latin-1 fold with an ISO-8859-1 wordlist run instead. */
	{"$secret$174432f8d71a1377566c9ffd521213297245d38f21f066a812a983d2f7e159d7", "abcdefghijklmno"},
	{"$secret$0f3b3a7d35378e8b0ae94895140032c4bf6a48b4cecfe4bb1d5f492ad3c67f63", "ABCDEFGHIJKLMNOPQRSTUVWX"},
	{"$secret$c87cd146a23eb9ecd07d09b1f556ca35", "abcdefghijklmn"},
	{NULL}
};

static int hex_decode(const char *src, unsigned char *dst, int max)
{
	int i, n = strlen(src) / 2;
	if (n < 1 || n > max)
		return -1;
	for (i = 0; i < n; i++) {
		unsigned char b = 0;
		int k;
		for (k = 0; k < 2; k++) {
			char c = src[i * 2 + k];
			b <<= 4;
			if (c >= '0' && c <= '9')      b |= c - '0';
			else if (c >= 'a' && c <= 'f') b |= c - 'a' + 10;
			else if (c >= 'A' && c <= 'F') b |= c - 'A' + 10;
			else return -1;
		}
		dst[i] = b;
	}
	return n;
}

static int secret_valid(char *ciphertext, struct fmt_main *self)
{
	unsigned char tmp[32];
	const char *p;
	int n;

	if (strncmp(ciphertext, FORMAT_TAG, FORMAT_TAG_LEN))
		return 0;
	p = ciphertext + FORMAT_TAG_LEN;
	if (strlen(p) != 32 && strlen(p) != 64) /* 16- or 32-byte blob */
		return 0;
	n = hex_decode(p, tmp, 32);
	if (n != 16 && n != 32)
		return 0;
	return 1;
}

static char *secret_split(char *ciphertext, int index, struct fmt_main *self)
{
	static char out[FORMAT_TAG_LEN + 64 + 1];

	snprintf(out, sizeof(out), "%s%s", FORMAT_TAG, ciphertext + FORMAT_TAG_LEN);
	return out;
}

static void *secret_get_salt(char *ciphertext)
{
	static struct custom_salt cs;
	memset(&cs, 0, sizeof(cs));
	cs.blob_len = hex_decode(ciphertext + FORMAT_TAG_LEN, cs.blob, 32);
	return &cs;
}

static void secret_set_salt(void *salt)
{
	cur_salt = (struct custom_salt *)salt;
}

static void secret_set_key(char *key, int index)
{
	strnzcpyn(saved_key[index], key, PLAINTEXT_LENGTH + 1);
}

static char *secret_get_key(int index)
{
	return saved_key[index];
}

static void init(struct fmt_main *self)
{
	init_tables();
	omp_autotune(self, OMP_SCALE);
	saved_key = mem_calloc_align(sizeof(*saved_key),
			self->params.max_keys_per_crypt, MEM_ALIGN_WORD);
	cracked = mem_calloc_align(sizeof(*cracked),
			self->params.max_keys_per_crypt, MEM_ALIGN_WORD);
	cracked_count = self->params.max_keys_per_crypt;
}

static void done(void)
{
	MEM_FREE(cracked);
	MEM_FREE(saved_key);
}

static int crypt_all(int *pcount, struct db_salt *salt)
{
	const int count = *pcount;
	int index;

	if (any_cracked) {
		memset(cracked, 0, sizeof(cracked[0]) * cracked_count);
		any_cracked = 0;
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (index = 0; index < count; index++) {
		unsigned char up[32];
		unsigned char key[16];
		unsigned char out[32];
		MD5_CTX ctx;
		AES_KEY aes_key;
		int len = strlen(saved_key[index]);
		int i, match;

		for (i = 0; i <= len; i++)
			up[i] = latin1_upper[(unsigned char)saved_key[index][i]];
		/*
		 * Expected plaintext = upper + 0x00 (addTrailingZero), then PKCS#5
		 * padding to 16 or 32 bytes: pad value 16 - ((len + 1) % 16), where
		 * a full block of padding is added when (len + 1) % 16 == 0.
		 */
		up[len] = 0x00;
		{
			int total = len + 1;
			int pad = 16 - (total % 16);
			memset(up + total, pad, 32 - total);
		}

		MD5_Init(&ctx);
		MD5_Update(&ctx, up, len);
		MD5_Final(key, &ctx);

		AES_set_decrypt_key(key, 128, &aes_key);
		AES_decrypt(cur_salt->blob, out, &aes_key); /* CBC with IV=0: P1 = D(C1) */
		match = !memcmp(out, up, 16);
		if (match && cur_salt->blob_len == 32) {
			/* verify the whole 32-byte blob including padding;
			 * CBC: P2 = D(C2) XOR C1 */
			unsigned char c1[16];
			int j;
			memcpy(c1, cur_salt->blob, 16);
			AES_decrypt(cur_salt->blob + 16, out + 16, &aes_key);
			for (j = 0; j < 16; j++)
				out[16 + j] ^= c1[j];
			match = !memcmp(out, up, 32);
		}
		cracked[index] = match;
#ifdef _OPENMP
#pragma omp atomic
#endif
		any_cracked |= match;
	}
	return count;
}

static int cmp_all(void *binary, int count)
{
	return any_cracked;
}

static int cmp_one(void *binary, int index)
{
	return cracked[index];
}

static int cmp_exact(char *source, int index)
{
	return 1;
}

struct fmt_main fmt_secret = {
	{
		FORMAT_LABEL,
		FORMAT_NAME,
		ALGORITHM_NAME,
		BENCHMARK_COMMENT,
		BENCHMARK_LENGTH,
		0,
		PLAINTEXT_LENGTH,
		BINARY_SIZE,
		BINARY_ALIGN,
		SALT_SIZE,
		SALT_ALIGN,
		MIN_KEYS_PER_CRYPT,
		MAX_KEYS_PER_CRYPT,
		FMT_8_BIT | FMT_OMP, /* case-INSENSITIVE: verifier is MD5(upper(pw)),
		                      * so "pw" and "PW" verify identically */
		{ NULL },
		{ FORMAT_TAG },
		secret_tests
	}, {
		init,
		done,
		fmt_default_reset,
		fmt_default_prepare,
		secret_valid,
		secret_split,
		fmt_default_binary,
		secret_get_salt,
		{ NULL }, /* tunable_cost_value */
		fmt_default_source,
		{
			fmt_default_binary_hash
		},
		fmt_default_salt_hash,
		NULL,
		secret_set_salt,
		secret_set_key,
		secret_get_key,
		fmt_default_clear_keys,
		crypt_all,
		{
			fmt_default_get_hash
		},
		cmp_all,
		cmp_one,
		cmp_exact
	}
};

#endif /* plugin stanza */
#endif /* HAVE_LIBCRYPTO || HAVE_COMMONCRYPTO */

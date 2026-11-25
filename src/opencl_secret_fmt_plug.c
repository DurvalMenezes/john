/*
 * OpenCL plugin for LinkeSOFT "Secret!" password manager
 * (Android/iPhone/BlackBerry/Desktop AES family), companion to
 * secret_fmt_plug.c and secret_kernel.cl.
 *
 * This software is Copyright (c) 2026 by Durval Menezes, and it is hereby
 * released to the general public under the terms of the WTF Program License
 * https://www.wtfpl.net/; There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 *
 * Algorithm reconstructed from com.linkesoft.secret.data.Crypt (Secret! for
 * Android 2.3.2, decompiled with androguard, 2026-11):
 *
 *   upper   = password.toUpperCase() in Latin-1 (ISO8859-1) bytes
 *   key     = MD5(upper)                          -> AES-128 key, no salt, no KDF
 *   verifier("encrpasswd") = AES-128-CBC-encrypt( key, IV = 16 zero bytes,
 *                                   plaintext = upper + 0x00, PKCS#5 padding )
 *
 * Input format: $secret$<32 or 64 hex chars> (the encrpasswd blob).
 *
 * GPU-side mask acceleration (FMT_MASK / mask_int_cand): up to
 * MASK_FMT_INT_PLHDR (4) of a --mask run's placeholder positions are
 * expanded on the device instead of on the host CPU. set_key()/get_key()
 * and the key_buf/key_idx transfer scheme, the int_key_loc/int_keys GPU
 * buffers, and the atomic-counter + out_index crack-reporting convention
 * are adapted line-for-line from opencl_o5logon_fmt_plug.c, john's only
 * other GPU-mask format and the verified reference for this mechanism.
 * Before this change, --mask runs were bound by john's single-threaded CPU
 * candidate generator feeding the GPU one fully-expanded password at a
 * time (README: ~26-36M p/s with the GPU ~2/3 idle, vs the ~113-130M c/s
 * wordlist/rules rate). With GPU-side mask expansion, the CPU only has to
 * generate and transfer one key per *outer* (non-GPU-offloaded) mask
 * combination; the device itself iterates NUM_INT_KEYS candidates per
 * work-item, so CPU-side candidate generation is no longer the bottleneck
 * for the offloaded positions. Without --mask, mask_int_cand.num_int_cand
 * is 1 and this degenerates to one iteration per work-item, same as the
 * pre-mask kernel plus the now-unavoidable key_buf/key_idx indirection
 * instead of a fixed stride.
 *
 * Autotune: the kernel does a single MD5 + AES per internal candidate,
 * i.e. without a mask it is far too fast for the autotuner to stop on its
 * duration limit: starting from 256 it would otherwise keep doubling GWS
 * (keys per crypt) until the per-key host buffer reaches multiple GB and
 * the benchmark's set_key() flood walks off the heap. reset() therefore
 * passes an explicit
 *   gws_limit = MIN(get_max_mem_alloc_size(gpu_id) / (2 * PLAINTEXT_LENGTH),
 *                    8M)
 * to both opencl_init_auto_setup() and autotune_run(), same safety margin
 * as before mask support was added.
 */

#ifdef HAVE_OPENCL

#define FORMAT_STRUCT fmt_opencl_secret

#if FMT_REGISTERS_H
john_register_one(&FORMAT_STRUCT);
#elif FMT_EXTERNS_H
extern struct fmt_main FORMAT_STRUCT;
#else

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>

#include "arch.h"
#include "opencl_common.h"
#include "mask_ext.h"
#include "opencl_helper_macros.h"
#include "options.h"

#define FORMAT_LABEL        "Secret-opencl"
#define FORMAT_NAME         ""
#define ALGORITHM_NAME      "MD5(pw) + AES-128-CBC(zero IV) OpenCL"
#define BENCHMARK_COMMENT   ""
#define BENCHMARK_LENGTH    0x107
#define PLAINTEXT_LENGTH    24
#define BINARY_SIZE         0
#define BINARY_ALIGN        1
#define SALT_SIZE           sizeof(struct secret_salt_cur)
#define SALT_ALIGN          4
#define MIN_KEYS_PER_CRYPT  1
#define MAX_KEYS_PER_CRYPT  1

#define FORMAT_TAG          "$secret$"
#define FORMAT_TAG_LEN      (sizeof(FORMAT_TAG) - 1)

/* Shared auto-tune stuff */
#define STEP                0
#define SEED                1024
#define ROUNDS              1

static const char *warn[] = {
	"pass xfer: ",  ", index xfer: ",  ", crypt: ",  ", res xfer: "
};

/* Must match the struct in secret_kernel.cl byte for byte. */
struct secret_salt_cur {
	uint32_t   blob_len;    /* 16 or 32 */
	unsigned char blob[32]; /* encrpasswd verifier, hex-decoded */
};

static char *key_buf;
static unsigned int *key_idx, key_buf_end;
static unsigned int crack_count_ret, *out_index;
static size_t key_offset, idx_offset;
static cl_mem cl_key_buf, cl_key_idx, cl_salt, cl_crack_count_ret, cl_out_index;
static cl_mem pinned_key_buf, pinned_key_idx, pinned_out_index;
static cl_mem pinned_saved_int_key_loc, cl_buffer_int_keys, cl_saved_int_key_loc;
static cl_uint *saved_int_key_loc;
static int static_gpu_locations[MASK_FMT_INT_PLHDR];
static const cl_uint zero = 0;

/*
 * Cached copy of the current salt, re-pushed to the GPU by create_clobj()
 * (see there for why: reset()'s own autotune_run() benchmarks using the
 * format's self-test vectors, which overwrites cl_salt with THEIR salts.
 * John core calls set_salt() with the real target exactly once, right
 * after the first reset() of a session - it does not anticipate a format
 * needing its salt re-applied after a LATER reset(), which is exactly
 * what happens here on every mask length change that alters int_mask_sum
 * (do_mask_crack() calls format->methods.reset() again per length as
 * needed). Without this cache, every length after the first one for which
 * reset() fires again crypt-checks candidates against a stale/self-test
 * salt instead of the real target - a silent false-negative, not a crash,
 * so it is easy to miss in testing that only exercises a single fixed
 * length.
 */
static struct secret_salt_cur cached_salt;
static int have_cached_salt;

static int new_keys;

extern struct fmt_main FORMAT_STRUCT;

#include "opencl_autotune.h" /* Must come after auto-tune definitions */

static size_t get_task_max_work_group_size()
{
	return autotune_get_task_max_work_group_size(FALSE, 0, crypt_kernel);
}

static void release_clobj(void);

static void create_clobj(size_t gws, struct fmt_main *self)
{
	release_clobj();

	CLCREATEPINNED(key_buf, CL_RO, PLAINTEXT_LENGTH * gws);
	CLCREATEPINNED(key_idx, CL_RO, sizeof(cl_uint) * (gws + 1));
	CLCREATEPINNED(out_index, CL_RW, sizeof(cl_uint) * gws * mask_int_cand.num_int_cand);
	CLCREATEBUFFER(cl_salt, CL_RO, SALT_SIZE);
	CLCREATEBUFFER(cl_crack_count_ret, CL_RW, sizeof(cl_uint));

	/* For GPU-side mask */
	CLCREATEPINNED(saved_int_key_loc, CL_RO, sizeof(cl_uint) * gws);
	CLCREATEBUFCOPY(cl_buffer_int_keys, CL_RO, 4 * mask_int_cand.num_int_cand,
	                mask_int_cand.int_cand ? mask_int_cand.int_cand : (void*)&zero);

	crack_count_ret = 0;
	CLWRITE(cl_crack_count_ret, CL_TRUE, 0, sizeof(cl_uint), &crack_count_ret, NULL);

	/* Re-push the real salt: see the comment on cached_salt above. */
	if (have_cached_salt)
		CLWRITE(cl_salt, CL_TRUE, 0, sizeof(struct secret_salt_cur) -
		        (32 - cached_salt.blob_len), &cached_salt, NULL);

	CLKERNELARG(crypt_kernel, 0, cl_key_buf);
	CLKERNELARG(crypt_kernel, 1, cl_key_idx);
	CLKERNELARG(crypt_kernel, 2, cl_salt);
	CLKERNELARG(crypt_kernel, 3, cl_crack_count_ret);
	CLKERNELARG(crypt_kernel, 4, cl_out_index);
	CLKERNELARG(crypt_kernel, 5, cl_saved_int_key_loc);
	CLKERNELARG(crypt_kernel, 6, cl_buffer_int_keys);
}

static void release_clobj(void)
{
	if (cl_salt) {
		CLRELEASEPINNED(out_index);
		CLRELEASEPINNED(key_buf);
		CLRELEASEPINNED(key_idx);
		CLRELEASEPINNED(saved_int_key_loc);
		CLRELEASEBUFFER(cl_crack_count_ret);
		CLRELEASEBUFFER(cl_salt);
		CLRELEASEBUFFER(cl_buffer_int_keys);
	}
}

static void done(void)
{
	if (program[gpu_id]) {
		release_clobj();

		HANDLE_CLERROR(clReleaseKernel(crypt_kernel), "Release kernel");
		HANDLE_CLERROR(clReleaseProgram(program[gpu_id]), "Release Program");

		crypt_kernel = NULL;
		program[gpu_id] = NULL;
	}
}

static void init(struct fmt_main *self)
{
	opencl_prepare_dev(gpu_id);

	/*
	 * Unlike o5logon's "tuned on 2080ti" comment, this divisor has not
	 * been empirically tuned for this format at all - it's carried over
	 * from the only reference GPU-mask format in the tree as a starting
	 * point. opencl_speed_index() scales with measured device speed, so
	 * it adapts somewhat automatically, but if --mask throughput looks
	 * off, this is the first knob to experiment with.
	 */
	mask_int_cand_target = opencl_speed_index(gpu_id) / 2048;
}

static void reset(struct db_main *db)
{
	char build_opts[256];
	size_t gws_limit;
	int i;

	if (crypt_kernel)
		done();

	gws_limit = MIN(get_max_mem_alloc_size(gpu_id) / (2 * PLAINTEXT_LENGTH),
	                8 * 1024 * 1024);

	for (i = 0; i < MASK_FMT_INT_PLHDR; i++)
		if (mask_skip_ranges && mask_skip_ranges[i] != -1)
			static_gpu_locations[i] = mask_int_cand.int_cpu_mask_ctx->
				ranges[mask_skip_ranges[i]].pos;
		else
			static_gpu_locations[i] = -1;

	snprintf(build_opts, sizeof(build_opts),
	         "-DPLAINTEXT_LENGTH=%u"
	         " -DLOC_0=%d"
#if MASK_FMT_INT_PLHDR > 1
	         " -DLOC_1=%d"
#endif
#if MASK_FMT_INT_PLHDR > 2
	         " -DLOC_2=%d"
#endif
#if MASK_FMT_INT_PLHDR > 3
	         " -DLOC_3=%d"
#endif
	         " -DNUM_INT_KEYS=%u -DIS_STATIC_GPU_MASK=%d",
	         PLAINTEXT_LENGTH,
	         static_gpu_locations[0],
#if MASK_FMT_INT_PLHDR > 1
	         static_gpu_locations[1],
#endif
#if MASK_FMT_INT_PLHDR > 2
	         static_gpu_locations[2],
#endif
#if MASK_FMT_INT_PLHDR > 3
	         static_gpu_locations[3],
#endif
	         mask_int_cand.num_int_cand, mask_gpu_is_static
		);

	if (!program[gpu_id])
		opencl_init("$JOHN/opencl/secret_kernel.cl", gpu_id, build_opts);

	/* create kernel to execute */
	if (!crypt_kernel)
		CLCREATEKERNEL(crypt_kernel, "secret");

	/* Initialize openCL tuning (library) for this format. */
	opencl_init_auto_setup(SEED, 0, NULL, warn, 2, &FORMAT_STRUCT,
	                       create_clobj, release_clobj, PLAINTEXT_LENGTH,
	                       gws_limit, db);

	/* Auto tune execution from shared/included code. */
	autotune_run(&FORMAT_STRUCT, 1, gws_limit, 200);
}

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

static struct fmt_tests secret_tests[] = {
	/* Test vectors generated from the reconstructed algorithm. */
	{"$secret$f7ba630de5803364d1891cfd02077fab", "cisco"},
	{"$secret$f3003d2aa438ad9306c5806b1916567c", "password"},
	{"$secret$9d83be34ca962e211cbe40f6b466869c", "secret"},
	{"$secret$24bd82c02a10825f59f307da07dcd50a", "PalmPilot1997"},
	{"$secret$51b3be590bf8eb5b9c072e7de4fc7456", "Tr0ub4dor&3"},
	{"$secret$cc9dce9431e3d9e0abd068afc56e9b0fa9b8347d7fc88fd882335aa0fd213ce8", "0123456789abcdef"},
	/* Edge cases: len=15 -> PKCS#7 pad 0x10 full block (32-byte blob);
	 * len=14 -> largest key with a 16-byte blob (pad = 1); len=24 =
	 * PLAINTEXT_LENGTH maximum. Identical to the CPU format's vectors
	 * (gen_vectors_edge.py). */
	{"$secret$174432f8d71a1377566c9ffd521213297245d38f21f066a812a983d2f7e159d7", "abcdefghijklmno"},
	{"$secret$0f3b3a7d35378e8b0ae94895140032c4bf6a48b4cecfe4bb1d5f492ad3c67f63", "ABCDEFGHIJKLMNOPQRSTUVWX"},
	{"$secret$c87cd146a23eb9ecd07d09b1f556ca35", "abcdefghijklmn"},
	{NULL}
};

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
	static struct secret_salt_cur cs;
	memset(&cs, 0, sizeof(cs));
	cs.blob_len = hex_decode(ciphertext + FORMAT_TAG_LEN, cs.blob, 32);
	return &cs;
}

/*
 * True if 'cs' is byte-for-byte one of this format's own self-test
 * vectors (secret_tests[] above), decoded once on first use.
 *
 * Needed because opencl_common.c's fill_opencl_device() - the internal
 * helper behind the GWS/LWS search inside EVERY reset()/autotune_run(),
 * not just the very first one - calls self->methods.set_salt() with
 * self->params.tests[0]'s salt ("cisco"'s verifier) purely to benchmark
 * timing. There is no existing flag (self_test_running/
 * bench_or_test_running cover only the top-level self-test/--test paths,
 * not this inner calibration call) that marks that call as synthetic, so
 * set_salt() has to recognize it by content instead: if it ever cached
 * THAT salt as "the real target", a later reset() (e.g. for a mask run's
 * next candidate length) would re-push the wrong verifier to the GPU and
 * every subsequent crypt_all() would silently check candidates against
 * someone else's hash - exactly the bug that made a known-correct "TEST"
 * password never crack when reached via --min-length=1.
 */
static int is_self_test_salt(const struct secret_salt_cur *cs)
{
	static unsigned char blobs[sizeof(secret_tests) / sizeof(secret_tests[0])][32];
	static unsigned int lens[sizeof(secret_tests) / sizeof(secret_tests[0])];
	static int built;
	int i;

	if (!built) {
		for (i = 0; secret_tests[i].ciphertext; i++)
			lens[i] = hex_decode(secret_tests[i].ciphertext + FORMAT_TAG_LEN,
			                      blobs[i], 32);
		built = 1;
	}

	for (i = 0; secret_tests[i].ciphertext; i++)
		if (cs->blob_len == lens[i] &&
		    !memcmp(cs->blob, blobs[i], lens[i]))
			return 1;
	return 0;
}

static void set_salt(void *salt)
{
	struct secret_salt_cur *cs = (struct secret_salt_cur *)salt;

	/*
	 * Only cache the REAL target's salt - see is_self_test_salt()'s
	 * comment for why this check exists at all.
	 */
	if (!is_self_test_salt(cs)) {
		cached_salt = *cs;
		have_cached_salt = 1;
	}

	CLWRITE(cl_salt, CL_FALSE, 0, sizeof(struct secret_salt_cur) -
	        (32 - cs->blob_len), cs, NULL);
	CLFLUSH();
}

static void clear_keys(void)
{
	key_buf_end = 0;
	key_idx[0] = 0;
	key_offset = 0;
	idx_offset = 0;
}

static void set_key(char *key, int index)
{
	if (mask_int_cand.num_int_cand > 1 && !mask_gpu_is_static) {
		int i;

		saved_int_key_loc[index] = 0;
		for (i = 0; i < MASK_FMT_INT_PLHDR; i++) {
			if (mask_skip_ranges[i] != -1)  {
				saved_int_key_loc[index] |= ((mask_int_cand.
				int_cpu_mask_ctx->ranges[mask_skip_ranges[i]].offset +
				mask_int_cand.int_cpu_mask_ctx->
				ranges[mask_skip_ranges[i]].pos) & 0xff) << (i << 3);
			}
			else
				saved_int_key_loc[index] |= 0x80 << (i << 3);
		}
	}

	while (*key)
		key_buf[key_buf_end++] = *key++;

	key_idx[index + 1] = key_buf_end;
	new_keys = 1;

	/* Early partial transfer to GPU */
	if (index && !(index & (256 * 1024 - 1))) {
		CLWRITE(cl_key_buf, CL_FALSE, key_offset, key_buf_end - key_offset, key_buf + key_offset, NULL);
		CLWRITE(cl_key_idx, CL_FALSE, idx_offset, 4 * (index + 2) - idx_offset, key_idx + (idx_offset / 4), NULL);

		if (!mask_gpu_is_static)
			CLWRITE(cl_saved_int_key_loc, CL_FALSE, idx_offset, 4 * (index + 1) - idx_offset, saved_int_key_loc + (idx_offset / 4), NULL);

		HANDLE_CLERROR(clFlush(queue[gpu_id]), "failed in clFlush");

		key_offset = key_buf_end;
		idx_offset = 4 * (index + 1);
		new_keys = 0;
	}
}

static char *get_key(int index)
{
	static char out[PLAINTEXT_LENGTH + 1];
	char *key;
	int i, len;
	int int_index = 0;

	if (crack_count_ret)
		index = out_index[index];
	if (mask_int_cand.num_int_cand) {
		int_index = index % mask_int_cand.num_int_cand;
		index /= mask_int_cand.num_int_cand;
	}

	key = &key_buf[key_idx[index]];
	len = key_idx[index + 1] - key_idx[index];

	for (i = 0; i < len; i++)
		out[i] = *key++;
	out[i] = 0;

	/* Re-apply GPU-side mask */
	if (len && mask_skip_ranges && mask_int_cand.num_int_cand > 1) {
		for (i = 0; i < MASK_FMT_INT_PLHDR && mask_skip_ranges[i] != -1; i++)
			if (mask_gpu_is_static)
				out[static_gpu_locations[i]] =
					mask_int_cand.int_cand[int_index].x[i];
			else
				out[(saved_int_key_loc[index] & (0xff << (i * 8))) >> (i * 8)] =
					mask_int_cand.int_cand[int_index].x[i];
	}

	return out;
}

static int crypt_all(int *pcount, struct db_salt *salt)
{
	int count = *pcount;
	size_t *lws = local_work_size ? &local_work_size : NULL;
	size_t gws = GET_NEXT_MULTIPLE(count, local_work_size);

	*pcount *= mask_int_cand.num_int_cand;

	if (new_keys) {
		/* Self-test kludge */
		if (idx_offset > 4 * (count + 1))
			idx_offset = 0;

		/* Safety for when count < GWS */
		for (int i = count; i <= gws; i++)
			key_idx[i] = key_buf_end;

		if (key_buf_end != key_offset)
			CLWRITE_CRYPT(cl_key_buf, CL_FALSE, key_offset, key_buf_end - key_offset, key_buf + key_offset, multi_profilingEvent[0]);
		CLWRITE_CRYPT(cl_key_idx, CL_FALSE, idx_offset, 4 * (gws + 1) - idx_offset, key_idx + (idx_offset / 4), multi_profilingEvent[1]);

		if (!mask_gpu_is_static)
			CLWRITE_CRYPT(cl_saved_int_key_loc, CL_FALSE, idx_offset, 4 * gws - idx_offset, saved_int_key_loc + (idx_offset / 4), NULL);

		new_keys = 0;
	}

	WAIT_INIT(gws)
	BENCH_CLERROR(clEnqueueNDRangeKernel(queue[gpu_id], crypt_kernel, 1, NULL, &gws, lws, 0, NULL, multi_profilingEvent[2]), "Failed running crypt kernel");
	CLFLUSH();
	/*
	 * WAIT_UPDATE must come AFTER the blocking wait, not before it:
	 * it measures (and wait_min remembers, over a 20-crypt window) how
	 * long this round actually took, so that WAIT_SLEEP next round can
	 * usleep() for ~97.7% of that duration instead of leaving the
	 * NVIDIA driver busy-spin the whole time inside the blocking read
	 * below (with AvoidBusyWait at its default of 1, that is the
	 * difference between ~100% and ~3% CPU on the feeder core with zero
	 * throughput cost - the GPU is already saturated).  tezos/pdf/nt
	 * et al. use this order; opencl_o5logon_fmt_plug.c, which this
	 * format's crypt_all() was adapted from, is the one format in the
	 * tree with WAIT_UPDATE before the wait, which makes wait_min
	 * collapse to zero so the sleep never engages.
	 */
	WAIT_SLEEP
	CLREAD_CRYPT(cl_crack_count_ret, CL_TRUE, 0, sizeof(cl_uint), &crack_count_ret, multi_profilingEvent[3]);
	WAIT_UPDATE
	WAIT_DONE

	if (crack_count_ret) {
		/*
		 * Benign - may happen when gws > count due to
		 * GET_NEXT_MULTIPLE(), particularly during self-test.
		 */
		if (crack_count_ret > *pcount)
			crack_count_ret = *pcount;

		CLREAD_CRYPT(cl_out_index, CL_TRUE, 0, sizeof(cl_uint) * crack_count_ret, out_index, NULL);

		CLWRITE_CRYPT(cl_crack_count_ret, CL_FALSE, 0, sizeof(cl_uint), &zero, NULL);
	}

	return crack_count_ret;
}

static int cmp_all(void *binary, int count)
{
	return count;
}

/*
 * If we got here, it's always a crack - but 'index' is a mapped one (see
 * o5logon_fmt_plug.c's comment on the same pattern). This is a salt-only
 * format with no binary, so there can only be one candidate per call that
 * matters; the subsequent get_key() call maps the out-index back and
 * re-applies the GPU-side mask to report which candidate cracked it.
 */
static int cmp_one(void *binary, int index)
{
	return crack_count_ret;
}

static int cmp_exact(char *source, int index)
{
	return 1;
}

struct fmt_main FORMAT_STRUCT = {
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
		FMT_8_BIT | FMT_MASK, /* case-INSENSITIVE: verifier is
		                       * MD5(upper(pw)), so "pw" and "PW" verify
		                       * identically (no FMT_CASE) */
		{ NULL },
		{ FORMAT_TAG },
		secret_tests
	}, {
		init,
		done,
		reset,
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
		set_salt,
		set_key,
		get_key,
		clear_keys,
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
#endif /* HAVE_OPENCL */

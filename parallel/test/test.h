#ifndef TEST_TEST_H_
#define TEST_TEST_H_

#include <stdio.h>

#define TEST_MAIN_INIT int __num_tests_failed = 0, __num_tests_run = 0;

#define RUN_TEST(func) { \
  if (!func()) { \
    fprintf(stderr, "TEST " #func " FAILED\n"); \
    __num_tests_failed++; \
  } \
  __num_tests_run++; \
}

#define TEST_INIT int __ret = 1;

#define TEST_RETURN return __ret;

#define ASSERT(func) { \
  if (!func()) { \
    fprintf(stderr, "ASSERT " #func " FAILED\n"); \
    __ret = 0; \
  } \
}

#endif  // TEST_TEST_H_

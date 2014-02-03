#include "src/integer.h"
#include "test/test.h"

bool test_integer_copy() {
   TEST_INIT

   integer num = integer_fromInt(500);
   integer num2;
   integer_copy(copy, num);
   ASSERT(integer_eq(num, num2))

   TEST_RETURN
}

void main() {
   TEST_MAIN_INIT


}

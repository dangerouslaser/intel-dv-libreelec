/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "dvbridge_fel_key_match.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
    int base[33], enhancement[33];
    assert(!dvbridge_fel_key_match(base, enhancement, 0));
    assert(!dvbridge_fel_key_match(base, enhancement, 33));
    for (unsigned count = 1; count <= 32; ++count) {
        for (unsigned i = 0; i < count; ++i) {
            base[i] = (int)i - 16;
            enhancement[count - i - 1] = base[i];
        }
        assert(dvbridge_fel_key_match(base, enhancement, count));
        int saved = enhancement[0];
        enhancement[0] = 1000;
        assert(!dvbridge_fel_key_match(base, enhancement, count));
        enhancement[0] = INT_MIN;
        assert(!dvbridge_fel_key_match(base, enhancement, count));
        enhancement[0] = saved;
        saved = base[0];
        base[0] = INT_MIN;
        assert(!dvbridge_fel_key_match(base, enhancement, count));
        base[0] = saved;
        if (count > 1) {
            enhancement[0] = enhancement[1];
            assert(!dvbridge_fel_key_match(base, enhancement, count));
            base[0] = base[1];
            assert(!dvbridge_fel_key_match(base, enhancement, count));
        }
    }
    puts("PASS: reordered picture sets, missing/duplicate pictures and 32-picture bound");
    return 0;
}

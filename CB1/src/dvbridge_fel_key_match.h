/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef DVBRIDGE_FEL_KEY_MATCH_H
#define DVBRIDGE_FEL_KEY_MATCH_H
#include <stdbool.h>
#include <limits.h>

/* A delayed key match must cover the same picture-order set in both layers. */
static bool dvbridge_fel_key_match(const int *base, const int *enhancement, unsigned count)
{
    if (!count || count > 32) return false;
    bool used[32] = {0};
    for (unsigned i = 0; i < count; ++i) {
        if (base[i] == INT_MIN) return false;
        for (unsigned k = 0; k < i; ++k)
            if (base[k] == base[i]) return false;
        unsigned j = 0;
        while (j < count && (used[j] || base[i] != enhancement[j])) ++j;
        if (j == count) return false;
        used[j] = true;
    }
    return true;
}
#endif

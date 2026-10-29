// overlay_source_colour (source_overlay.cl, FP_CONTRACT OFF) over `coeff[]`.
// overlay_matrix() / overlay_source_colour() (source_overlay.cl, FP_CONTRACT OFF).
vec3 overlay_matrix(uint m, vec3 v)
{
    precise vec3 r;
    r.x = coeff[m + 0] * v.x + coeff[m + 1] * v.y + coeff[m + 2] * v.z;
    r.y = coeff[m + 3] * v.x + coeff[m + 4] * v.y + coeff[m + 5] * v.z;
    r.z = coeff[m + 6] * v.x + coeff[m + 7] * v.y + coeff[m + 8] * v.z;
    return r;
}

bool overlay_source_colour(uvec4 p, out vec4 result)
{
    result = vec4(0.0);
    if (p.w == 0u)
        return true;
    if (any(greaterThan(p.xyz, uvec3(p.w))))
        return false;
    precise vec3 s = vec3(p.xyz) / float(p.w);
    precise vec3 decoded = pow((s + 0.055) / 1.055, vec3(2.4));
    precise vec3 lin = mix(decoded, s / 12.92, lessThanEqual(s, vec3(0.04045))) * (coeff[30] / 10000.0);
    vec3 common_lms = overlay_matrix(18u, lin);
    vec3 source_linear = overlay_matrix(9u, common_lms);
    if (any(isnan(source_linear)) || any(isinf(source_linear)) || any(lessThan(source_linear, vec3(0.0))) ||
        any(greaterThan(source_linear, vec3(1.0))))
        return false;
    precise vec3 power = pow(source_linear, vec3(2610.0 / 16384.0));
    precise vec3 pq = pow((3424.0 / 4096.0 + (2413.0 / 128.0) * power) / (1.0 + (2392.0 / 128.0) * power),
                          vec3(2523.0 / 32.0));
    precise vec3 codes = (overlay_matrix(0u, pq) + vec3(coeff[27], coeff[28], coeff[29])) * 4096.0;
    if (any(isnan(codes)) || any(isinf(codes)) || any(lessThan(codes, vec3(0.0))) || any(greaterThan(codes, vec3(4095.0))))
        return false;
    result = vec4(codes, float(p.w) / 255.0);
    return true;
}


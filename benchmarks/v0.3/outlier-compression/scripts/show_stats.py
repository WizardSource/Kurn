import json, sys
s = json.load(open(sys.argv[1]))
L = len(s["layers"]) - 1
print("resid: layer top5share(nonfirst) top5share(all) firstTokAbsmax@dim firstTok/other energy cv(top5) maxabs_nonfirst rms")
for i, l in enumerate(s["layers"][:L]):
    r = l["resid"]
    print(i, f'{r["top5_energy_share_nonfirst"]:.2f} {r["top5_energy_share_all"]:.3f} {r["first_tok_absmax"]:.0f}@{r["first_tok_argmax"]} {r["first_tok_energy_x_nonfirst"]:.0f}x',
          "cv", [round(c, 2) for c in r["top5_cv"]], "mean", [round(c, 1) for c in r["top5_mean"]], f'{r["max_abs_nonfirst"]:.0f} {r["rms_nonfirst"]:.2f}')
print("sites: layer site top16share absmax/rms(med) absmax/rms(first) maxabs | relerr bf16 g32 tok tok+out16 tok+shift tok+shift+out16")
for i, l in enumerate(s["layers"]):
    for n, v in l["sites"].items():
        print(i, n, f'{v["top16_energy_share"]:.2f} {v["absmax_over_rms_median"]:.1f} {v["absmax_over_rms_first_tok"]:.0f} {v["max_abs"]:.0f} | '
              f'{v["relerr_bf16"]:.4f} {v["relerr_g32"]:.4f} {v["relerr_tok"]:.4f} {v["relerr_tok_out16"]:.4f} {v["relerr_tok_shift"]:.4f} {v["relerr_tok_shift_out16"]:.4f}', v["top16"][:6])

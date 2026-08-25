# PNAS LaTeX template provenance

The manuscript and SI use the current official PNAS Overleaf templates that
were available on 2026-08-25:

- Research Report (2025):
  <https://www.overleaf.com/latex/templates/template-for-preparing-your-research-report-submission-to-pnas-using-overleaf-2025/whbdryzwztnd>
- Supporting Information:
  <https://www.overleaf.com/latex/templates/pnas-template-for-supplementary-information/wqfsfqwyjtsd>

`Research_report_2025.tex` and `PNAS_SI_2025.tex` are the unmodified template
entry points retrieved from those pages. The current class, style,
bibliography style, and logo were recovered from a pinned public export of the
official bundle at
<https://github.com/kr-colab/cxt/tree/294b0c27226accbac830c9c922de6b2f92f83fe8/cxt_paper/cxt_maintext>
(retrieved 2026-08-25). The installed class identifies itself as
`pnas-new.cls` version 1.47. The unchanged `jabbrv` files are also identical to
the earlier official-template mirror at
<https://github.com/alcrene/myst-template-pnas2023>.

SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `Research_report_2025.tex` | `fa3d38d01f4c6e60055a06c55168cc3424647c2e22b91fb4274f2934bb973917` |
| `PNAS_SI_2025.tex` | `02c643f37f453dc68b2e5b9e0bd1567eed4ed00404cc19f8e477c64426aeb7c0` |
| `pnas-new.cls` | `fd090d07842ac379663ff54d02cc3f532e9c4e2aa9b09ddb243874706aaca99b` |
| `pnasresearcharticle.sty` | `92c7be2586e72aef81342fff47a8dce94f78fdc500ef619f12e25fbbb00d9ccf` |
| `pnas-new.bst` | `26e4afeccebbe5e70bbe60f756504aae2a498aac78181231a446857024b2a5c9` |
| `PNAS_Logo.pdf` | `32264cf1f6c84f60d635943727eca7e6b08fe5972673b5bead95322d3c92ac36` |
| `pnassupportinginfo.sty` | `aa4541ff05f3aa54a8f66ae0d0a223a7fecc5dfe528d995c1c03c2a5538356a3` |
| `pnas-logo.png` | `8b25ba9dbce328e735f2bc979a0688363f66e7f274b422a731f2edb9f058df5e` |
| `jabbrv.sty` | `79110f7efbc1860f7b12477c2f1ae36edf3e44922203dff820a4fef3cc5de4d1` |
| `jabbrv-ltwa-all.ldf` | `af218569e01a300ec7ddceabfba04fefd9a826c3759393ddcd4ac894bc904ea5` |
| `jabbrv-ltwa-en.ldf` | `a6ec2fbd6a397adcbbc4de0bdb1acf6b35dfb0ee7a558a6bf66c50c96750f201` |

Build commands set `TEXINPUTS=template//:` so the journal support files remain
isolated from the manuscript source.

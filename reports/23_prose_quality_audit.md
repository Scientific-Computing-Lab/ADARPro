# 23 — Final prose-quality audit

## Sources and editorial hierarchy

This pass uses the complete Academic Humanizer instructions at revision
`94b88b23703bed7df507acae7d6d5876209a0cdf` and the complete Human-Patterned
Language prompt at revision `a8b13dbc610d10d035837970d60c59687390c86a`.
Academic Humanizer governs scholarly register, claim–evidence alignment, and
preservation of numbers, citations, and technical terms. HPL contributes its
diagnosis of uniform rhythm, repeated discourse moves, nominalization,
antithesis, and over-explicit transitions. Its recommendations to add
fragments, contractions, rhetorical questions, informal transitions,
emotional language, or deliberate roughness are inappropriate for a PNAS
paper and are not applied.

## Pre-edit pattern inventory

1. **Dense repetition of the organizing vocabulary.** `information` appears
   41 times in the main source, alongside repeated uses of `recover`,
   `recoverable`, `information map`, and `information scale`. These terms are
   necessary in the title and central definitions but often stand where a
   concrete entity or result would be clearer.
2. **Uniform result-section cadence.** Several subsections open with a broad
   conclusion, proceed through experiment and numbers, and end with a
   formulaic interpretive sentence. The structure is logical but becomes
   predictable across successive sections.
3. **Repeated rhetorical pivots.** The manuscript uses `therefore`, `thus`,
   `consequently`, `likewise`, and `rather than` more often than the reasoning
   requires. Several positive claims are framed through a rejected alternative
   (`not X`, `rather than X`) even when the contrast is already clear.
4. **List-heavy sentence architecture.** The Significance Statement,
   Abstract, final Introduction paragraph, and Discussion repeatedly use
   balanced three-part clauses. Some lists are biologically necessary, but
   their concentration at rhetorical boundaries makes the prose sound
   composed to a template.
5. **Clause stacking.** Long sentences in the Abstract, target-local Results,
   held-out Results, Discussion, and Methods perform two or three logical jobs.
   Splitting selected sentences should improve reading without making the
   paper terse.
6. **Abstract containers and nominalizations.** Phrases such as `framework`,
   `present recoverability boundary`, `operational validation`, `conditional
   saturation`, and `final predictive specification` sometimes obscure the
   measured comparison. Concrete subjects and verbs can replace them.
7. **Boilerplate inference frames.** `These results establish`, `This result
   confirms`, `This map connects`, `The remaining boundary is`, and `The
   broader result is` announce interpretation before delivering it. Most can
   be replaced by the interpretation itself.
8. **Overstrong verbs at a few evidence boundaries.** `Establish` is sometimes
   used for descriptive cross-dataset performance or development-stage
   attribution. Bounded verbs such as `shows`, `supports`, or a direct report
   of the measured result are more exact.
9. **Em-dash constructions.** LaTeX triple hyphens occur in explanatory
   parentheticals and one freeze statement. Parentheses, commas, or separate
   sentences will read more cleanly.
10. **Caption density.** Figure captions accurately preserve scope, but some
    titles and final sentences carry both interpretation and evidential
    qualification. Shortening these will make the visual argument easier to
    scan without dropping safeguards.

## Non-negotiable preservation checks

- Keep every quantitative macro, literal result, equation, threshold, sample
  count, citation key, figure/table reference, and endpoint definition.
- Preserve the distinction between development OOF attribution and frozen
  held-out confirmation.
- Preserve the noncausal interpretation of tissue context and target-centered
  pairing, including the inability to isolate partner identity.
- Preserve the conditional scope of saturation and the lack of an independent
  human-cohort predictor evaluation.
- Leave stable technical names unchanged where synonym variation would reduce
  precision.

## Post-edit verification

- The main source decreased from 3,793 to 3,618 whitespace-delimited tokens
  (4.6%) without removing a result or evidence boundary.
- Uses of `information` decreased from 41 to 13. The remaining instances name
  the paper's central construct or a specific model input.
- Formulaic inference and transition phrases were reduced without adding
  casual prose, fragments, contractions, rhetorical questions, or decorative
  variation.
- Sorted citation calls, quantitative macro calls, literal numerical tokens,
  and figure/table reference calls are byte-identical to the pre-edit source.
- Development OOF evidence, frozen held-out confirmation, noncausal tissue and
  pairing interpretations, conditional saturation, and the external-cohort
  limitation retain their original epistemic scope.

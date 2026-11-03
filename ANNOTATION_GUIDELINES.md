# Annotation guidelines (draft v0.1)

These are provisional guidelines for the starter dataset. They should be revised after a pilot annotation round with at least two annotators, and agreement (e.g. Cohen's kappa) should be reported.

## Unit of annotation

One row = one sentence containing **one target marker**. Label the *function of the marker in that sentence*, not the marker in general.

## Labels

**consequence** – the second segment is a result or effect of the first.
Test: can you paraphrase with "because of that" or "as a result"?
> Il a plu toute la nuit, **donc** les routes sont glissantes.

**contrast** – the second segment opposes, limits, or runs against an expectation created by the first.
Test: can you paraphrase with "but" or "despite that"?
> Le plan semble solide ; **cependant**, il présente des risques.

**reformulation** – the second segment restates, specifies, or summarises the first in other words.
Test: can you paraphrase with "in other words" or "that is"?
> Elle est bilingue, **c'est-à-dire** qu'elle parle deux langues.

## Rules

1. One marker per sentence; keep the marker's spelling identical in the `marker` and `sentence` columns (the code looks for an exact match, ignoring case).
2. Use straight apostrophes (`'`) throughout.
3. Do not label a sentence if the marker has a non-discourse use (e.g. temporal *alors* "at that time", manner *ainsi* "in this way", disjunctive *soit ... soit*). Leave it out or add a separate label later.
4. If two functions seem plausible, record the sentence in a separate "ambiguous" file instead of forcing a label.

## Known problem cases (decide before scaling up)

- **en fait** – can signal contrast, reformulation/correction, or mere emphasis. Deliberately **excluded** from the starter set.
- **alors**, **ainsi**, **donc** – each has non-consequence uses; check every sentence.
- **bref**, **en somme**, **en un mot** – labelled *reformulation* here, but a "summary/conclusion" label may be more appropriate.
- **autrement dit** – can sound like a consequence ("so"); the starter examples were written to favour the restatement reading.
- **au contraire** – sometimes a correction rather than a plain contrast.

## Extending the dataset

- Replace constructed sentences with naturally occurring ones from a documented source (corpus, licence, sampling method).
- Keep several different markers per label so that models cannot rely on one word.
- Record annotator IDs and disagreements; keep adjudicated labels separate from raw labels.

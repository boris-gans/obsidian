---
tags: [concept]
courses: [NLP]
sources:
  - course: NLP
    file: Session 07 - Information Retrieval.pdf
created: 2026-05-02
---

# Relevance feedback

A technique that refines query representation **after an initial ranking** by using user feedback. Documents marked relevant **reinforce** certain terms; non-relevant documents **reduce** their influence. The query effectively evolves into a better statistical representation of the user's information need.

## Rocchio update equation

$$\vec{q}_{i+1} = \vec{q}_i + \frac{\beta}{R}\sum_{j=1}^R \vec{r}_j - \frac{\gamma}{S}\sum_{j=1}^S \vec{s}_j$$

| Symbol | Meaning |
|---|---|
| $\vec{q}_i$ | original query vector |
| $\vec{q}_{i+1}$ | updated query vector |
| $\vec{r}_j$ | relevant documents from initial ranking |
| $\vec{s}_j$ | non-relevant documents |
| $R, S$ | counts of relevant / non-relevant |
| $\beta$ | how far to push toward relevant docs |
| $\gamma$ | how far to push away from non-relevant docs |

The update **shifts the query** toward the centroid of relevant documents and away from the centroid of non-relevant ones.

## What it shows about relevance

> Relevance in Information Retrieval is **not static**. It is an adaptive, comparative notion that depends on user interaction and the structure of the remaining document collection.

This **residual collection** view — what documents remain after marking some as relevant / non-relevant — is a precursor to interactive search and to modern **active learning** loops.

## Why it preceded modern dense retrieval

The Rocchio update is essentially **vector arithmetic in TF-IDF space** — a precursor to the operations used in dense retrieval today (e.g. moving a query embedding toward positive examples). The same idea, in dense embedding space, drives modern retrieval-based systems and recommendation feedback loops.

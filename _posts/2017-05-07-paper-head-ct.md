---
layout: post
title: "Research Paper: \"Artificial neural networks: Predicting head CT findings in elderly patients presenting with minor head injury after a fall\""
math: true
style: |
    .center_container {
        text-align:center;
        margin-bottom: 1em;
    }

    .paper {
        width: 100%;
        border: 1px solid #AAA
    }
#excerpt:
# Note: Create paper pdf image with:
#   `montage paper-head_ct.pdf[0-3,5,7-10] -mode Concatenate -tile x1 -quality 100 -resize x254 paper-head_ct.jpg && convert paper-head_ct.jpg -border 1 paper-head_ct.jpg`
---

<div class="center_container">
  <a href="{{ site.baseurl }}/assets/papers/paper-head_ct.pdf">
    <img class="paper" src="{{ site.baseurl }}/assets/papers/paper-head_ct.jpg" alt="Head CT Paper"/>
  </a>
</div>

After completing my first year of medical school in 2013, I started working on a medical research project during a summer research program hosted at the Brody School of Medicine.  The goal was simple: predict computed tomography (CT) findings of elderly patients presenting to the emergency department (ED) with minor head injury after a fall.  No problem, right?  Let’s unpack this a bit.  The scenario here is that an elderly patient age ≥ 65 sustains a fall with minor injury to the head, and is rushed off to the nearest ED for evaluation.  Based on current clinical decision rules used in Emergency Medicine (EM), this patient would automatically receive a head CT to look for any intracranial injuries (i.e. bleeding within the brain), which could be potentially life-threatening.  However, it turns out that only ~2% of these patients actually end up having any pathological findings.  That means that ~98% of these patients receive unnecessary head CTs, which are both costly (millions of dollars per year in the United States alone!) and risky due to the radiation exposure.  Thus the project: how could we extend the current decision rules to further stratify this group of patients into those who would and would not benefit from a head CT while still maintaining a high level of care?

The following paper published in the American Journal of Emergency Medicine (AJEM) describes our work using neural networks* for a machine learning approach to this problem.

<div class="center_container">
  <h4><a href="{{ site.baseurl }}/assets/papers/paper-head_ct.pdf">Accepted Preprint</a> | <a href="https://authors.elsevier.com/a/1UnGl_GkZ8~Lgy">Published Paper (AJEM 2017)</a></h4>
</div>

\* (Note: In the paper we use the older term “artificial neural networks”, rather than the usual "neural networks", to distinguish from biological neural networks for a medically-focused reader.)


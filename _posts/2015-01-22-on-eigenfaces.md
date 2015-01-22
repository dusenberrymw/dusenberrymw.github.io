---
layout: post
title: "On Eigenfaces: Creating ghost-like images from a set of faces."
math: true
style: |
    .faces_container {
        text-align:center;
        margin-bottom: 1em;
    }

    .faces {
        height:24em;
        width: 24em;
    }
#excerpt: Eigenfaces are fascinating!
---

<div class="faces_container">
    <img class="faces" src="{{ site.baseurl }}/assets/images/eigenfaces_sample.jpg" alt="Eigenfaces"/>
</div>

_**Eigenfaces**_ are super interesting extensions to the concept of eigenvectors, and also serve as fascinating visualizations.  Basically, if you have a large dataset of face images, eigenfaces are a set of face-like images that collectively capture the variation in the original set of faces.  Given a subset of the eigenfaces, an original face can be encoded as relative amounts of each eigenface, and then rebuilt by adding the encoded amounts of each eigenface  together. 

Let's explore these in more detail! 

### The Dataset
Say you have a large dataset of faces and wish to find these eigenfaces.  We'll use a [cropped version](http://conradsanderson.id.au/lfwcrop/) of the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset that will contain 5000 example faces, each 32px by 32px (1024 pixels total per face), and saved in the grayscale format. Let's visualize the first 100 faces:

<div class="faces_container">
    <img class="faces" src="{{ site.baseurl }}/assets/images/faces_original.jpg" alt="Original Faces"/>
</div>

As stated above, each image has $$n = 1024$$ pixels, and can be stored as a vector $$x^{(i)} \in \mathbb{R}^{1024}$$.  For the $$m = 5000$$ images in our dataset, we can store all of the images in a matrix $$X \in \mathbb{R}^{5000 \times 1024}$$; one row for each image vector $$x^{(i)}$$.

### Vector Spaces
In our dataset, the images lie in a 1024-dimensional vector space, which can be considered fairly high-dimensional.  However, since images of faces share many characteristics, the images are likely to be clustered together in the 1024-dimensional space.  Therefore, it would seem plausible that a lower dimensional vector subspace could still accurately represent the images, while preserving most of the variability in the original data.

The **goal** then becomes finding the $$k$$-dimensional subspace in which the data approximately lies.

In order to do this, we will start by finding a direction $$u$$ that will form a 1-dimensional subspace that approximates the data with the most variability possible.  Concretely, if we have an $$n$$-dimensional vector $$u$$, we can project an $$n$$-dimensional image $$x^{(i)}$$ onto it by $$x^Tu$$, resulting in a 1-dimensional, scalar value.  The variability of the set of $$m$$ projected images can then be calculated as $$\sum_{i=1}^m ({x^{(i)}}^Tu)^2$$.  For our matrix of images, the projections can be written as $$Xu$$, and the variance written more concisely as the squared norm of the projections, $$\|Xu\|^2$$.

As stated above, we want to find a direction $$u$$ that preserves the most variance.  Thus we want to maximize the projection variance $$\sigma^2$$, subject to $$u$$ being unit-length ($$\|u\| = 1$$) since we only want a direction.  If we rewrite the equation in a certain way, we can discover some clever math:

$$
\begin{aligned}
    \sum_{i=1}^m ({x^{(i)}}^Tu)^2                       =& \ \sigma^2 \\
    \sum_{i=1}^m u^Tx^{(i)}{x^{(i)}}^Tu                 =& \ \sigma^2 \\
    u^T \left(\sum_{i=1}^m x^{(i)}{x^{(i)}}^T \right)u  =& \ \sigma^2 \\
    u^T (X^TX)u                                         =& \ \sigma^2 \\
    (X^TX)u                                             =& \ \sigma^2u 
\end{aligned}
$$

Notice that in the final line, a large matrix $$X^TX$$ multiplied by the vector $$u$$ becomes equal to a scalar variance $$\sigma^2$$ multiplied by the same $$u$$ vector.  That equality leads us to the concept of eigenvectors & eigenvalues, which will help us to maximize these equations w.r.t. the variance.

### Eigenvectors & Eigenvalues
An **eigenvector** is a vector $$v$$ that when multiplied by a matrix $$A$$, becomes equal to itself multiplied by a scalar **eigenvalue** $$\lambda$$.

$$
Av = \lambda v
$$

Notice the correlation between this definition and the rewritten equation from above.  Let's write them out again:

$$
\begin{aligned}
    Av =& \lambda v \\
    (X^TX)u =& \sigma^2u 
\end{aligned}
$$

From this we can see that our $$X^TX$$ matrix maps to the matrix $$A$$ (and is known as the *covariance* matrix of $$X$$), $$u$$ is the eigenvector $$v$$, and the variance $$\sigma^2$$ is the eigenvalue $$\lambda$$.  So, with a bit of clever rewriting, we were able to redefine our goal in terms of eigenvectors/eigenvalues, which can be computed using existing algorithms.
 
### Goal, continued
With our problem redefined, we can now compute the vector $$u$$ onto which to approximate the data with the most variability by solving for the eigenvectors & eigenvalues of of the covariance matrix, $$X^TX$$.  If we assume that the eigenvectors have length 1 (we assumed $$\|u\| = 1$$), then we can basically assume there will be one eigenvector per eigenvalue (technically both $$v$$ and $$-v$$ will still be valid eigenvectors with the same eigenvalue, but that's okay).  Now we have a matrix $$U$$ consisting of the $$u^1, ..., u^n$$ eigenvectors, and a matrix $$\Lambda$$ consisting of the associated $$\lambda^1, ..., \lambda^n$$ eigenvalues.

So, if we solved for the eigenvectors and eigenvalues, the $$u$$ that maximizes the projection variance will be the **eigenvector with the largest associated eigenvalue**.

### K-dimensional subspace
Now that we know how to find a vector $$u$$ that creates a 1-dimensional subspace to project the data into with the most variance, finding $$k$$ vectors to create a $$k$$-dimensional subspace with the most variance possible is trivial.  Simply select the top $$k$$ eigenvectors with the highest associated eigenvalues.

### PCA
The algorithm we have discussed so far is known as **Principal Component Analysis**, or PCA.  It allows for determining the *prinicipal components* (the eigenvectors) that capture the most variance in the original data when projecting into the subspace created by them.  The idea of PCA is that it allows for an $$n$$-dimensional dataset to be represented in a reduced set of $$k$$-dimensions, and thus is also known as a *dimensionality reduction* algorithm.

### Eigenfaces
It is now time to discuss **eigenfaces**.  As discussed above, the $$U$$ eigenvectors can be considered the principal components of a dataset, and each capture a certain amount of the variability within the original data.  

As we also mentioned above, if each image is $$n$$-dimensional ($$n$$ pixels), then each eigenvector $$u$$ will also be $$n$$-dimensional.  Each value $$u_i$$ is computed to capture the most variation at the $$i$$th pixel location across the dataset.

Therefore, if we visualize each $$u$$ as if it were an image, we will find that they **appear as _ghost-like faces_**.  Let's plot the top 36 eigenfaces for our dataset:

<div class="faces_container">
    <img class="faces" src="{{ site.baseurl }}/assets/images/eigenfaces.jpg" alt="Eigenfaces"/>
</div>

Fascinating!

### Projection & Rebuilding
Now that we have the eigenfaces, we can project our original faces onto a subset of $$k$$ of them, thus reducing each image from $$n$$-dimensions down to a vector $$z$$ of $$k$$-dimensions.

$$
z^{(i)} =
\begin{bmatrix}
{x^{(i)}}^Tu^1 \\
{x^{(i)}}^Tu^2 \\
\vdots \\
{x^{(i)}}^Tu^k \\
\end{bmatrix}
\in \mathbb{R}^{k}
$$

The images can then be rebuilt in $$n$$ dimensions from the $$z$$ encodings, with some loss in accuracy, using the following:

$$
x^{(i)}_{\text{approx}} = (z^{(i)}u^1) + \ ... \ + (z^{(i)}u^k)
$$

Let's project our data onto $$k = 100$$ eigenfaces, and then rebuild the original faces using the eigenfaces and $$z$$ encodings for each image, resulting in the images below:

<div class="faces_container">
    <img class="faces" src="{{ site.baseurl }}/assets/images/faces_recovered.jpg" alt="Recovered Faces"/>
</div>

If we compare them side by side, we should see that the rebuilt images were close to the originals:

<div class="faces_container">
    <img src="{{ site.baseurl }}/assets/images/faces_comparison.jpg" alt="Faces Comparison"/>
</div>

So, while each original image had 1024 pixels, by using only 100 eigenface encodings for each image, we were able to rebuild each image with pretty good accuracy.  Awesome!

### Final thoughts
Overall, eigenfaces are super interesting.  Hopefully this post serves as an interesting guide to them, and sparks interest into further exploration on the topic!



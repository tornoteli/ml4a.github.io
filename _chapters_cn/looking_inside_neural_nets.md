---
layout: chapter
title: "Looking inside neural nets 深入神经网络内部"
includes: [mathjax, jquery, convnetjs, dataset, convnet, visualizer]
header_image: "/images/headers/brainbow.jpg"
header_quote: "lovelace"
---

<!--
brainbow by katie matho

http://medicalxpress.com/news/2015-10-brain-cells.html img.medicalxpress.com/newman/gfx/news/hires/2015/1-researchersl.png

http://journal.frontiersin.org/article/10.3389/fnana.2014.00103/full

http://catalog.flatworldknowledge.com/bookhub/reader/127?e=stangor-ch03_s01 images.flatworldknowledge.com/stangor/stangor-fig03_003.jpg
[printblock analogy?] [borges quote about numbers]
-->

In the [previous chapter](/ml4a/neural_networks), we saw how a neural network can be trained to classify handwritten digits with a respectable accuracy of around 90%. In this chapter, we are going to evaluate its performance a little more carefully, as well as examine its internal state to develop a few intuitions about what's really going on. Later in this chapter, we are going to break our neural network altogether by attempting to train it on a more complicated dataset of objects like dogs, automobiles, and ships, to try to anticipate what kinds of innovations will be necessary to take it to the next level.

在[上一章](/ml4a/cn/neural_networks)，我们看到了如何训练神经网络，以使其分辨手写数字的精度达到相当可观的90%。在本章，我们将进一步分析它的表现，同时深入它的内部状态（internal state）以直观了解到底发生了什么。在本章晚些时候，我们会通过在更复杂的数据集例如狗、汽车、船舶上训练我们的神经网络，来打破原有的神经网络，并且做一些新的尝试来让它更上一层楼。
 
## Visualizing weights 将权重可视化

Let's take a network trained to classify MNIST handwritten digits, except unlike in the last chapter, we will map directly from the input layer to the output layer with no hidden layers in between. Thus our network looks like this.

让我们先拿来一个用来分类 MNIST 手写数字的的神经网络，不过和上一章有一点不同，我们会直接从输入层（input layer）映射到输出层（output layer），而不经过隐藏层（hidden layer）。然后我们的神经网络看起来是这样。

{% include figure.html path="/images/figures/mnist_1layer.png" caption="1-layer neural network for MNIST" %}

{% include todo.html note="label output neurons" %}

Recall that when we input an image into our neural net, we visualize the network diagram by "unrolling" the pixels into a single column of neurons, as shown in the below figure on the left. Let's focus on just the connections plugged into the first output neuron, which we will label $$z$$, and label each of the input neurons and their corresponding weights as $$x_i$$ and $$w_i$$.

回想当我们输入一张图像到神经网络里，我们是这样可视化这个过程的：我们“展开”所有像素，然后放进一列神经元中，如下方左图所示。我们先来看第一个神经元，设为 $$z$$。这张图片的每个像素依次设为 $$x_i$$，它们的权重分别设为 $$w_i$$。 

{% include figure.html path="/images/figures/weights_analogy_1.png" caption="Highlighting the weights connections to a single output neuron" %}

{% include todo.html note="label $z$ on the left" %}

Rather than unrolling the pixels though, let's instead view the weights as a 28x28 grid where the weights are arranged exactly like their corresponding pixels. The representation on the above right looks different from the one in the below figure, but they are expressing the same equation, namely $$z=b+\sum{w x}$$.

比起把所有的像素展开成一条线，我们不如把所有的权重视为一个 28x28 的网格，然后将所有的权重按照相应的像素依次排列。上方右图可能和下图看起来不太相像，但是它们表达的是同一个方程：$$z=b+\sum{w x}$$。

{% include figure.html path="/images/figures/weights_analogy_2.png" caption="Another way to visualize the pixel-weights multiplication for each output neuron" %}

Now let's take a trained neural network with this architecture, and visualize the learned weights feeding into the first output neuron, which is the one responsible for classifying the digit 0. We color-code them so the lowest weight is black, and the highest is white.

现在我们拿一个这样的神经网络，并且把传输给第一个输出神经元的权重可视化，也就是负责分类数字 0 的那个。我们用不同颜色标记它们，使最低的权重对应黑色，反之对应白色。

{% include figure.html path="/images/figures/rolled_weights_mnist_0.png" caption="Visualizing the weights for the 0-neuron of an MNIST classifier" %}

Squint your eyes a bit... does it look a bit like a blurry 0? The reason why it appears this way becomes more clear if we think about what that neuron is doing. Because it is "responsible" for classifying 0s, its goal is to output a high value for 0s and a low value for non-0s. It can get high outputs for 0s by having large weights aligned to pixels which _tend_ to usually be high in images of 0s. 
Simultaneously, it can obtain relatively low outputs for non-0s by having small weights aligned to pixels which tend to be high in images of non-0s and low in images of 0s. The relatively black center of the weights image comes from the fact that images of 0s tend to be off here (the hole inside the 0), but are usually higher for the other digits.

眯起你的眼睛看看...它看起来像不像是一个模糊的 0？ 如果我们想一想这个神经元在做什么，就会更清楚它为什么看起来是这样。因为它是负责来分类所有 0 的图片，它的目标是为是 0 的图片输出一个较大值，为不是 0 的输出一个较小值。它可以通过给 0 的图片赋予一个较大的权重来得到较大的输出值，这个权重会**倾向于**给是 0 的图片取较大的值。同时，他会通过赋予较低的权重，为不是 0 的图片取相对低的值。图中有相对发黑的中心区域（0 中间的洞）是因为 0 的图片通常不经过那里，但是这个区域对于其他数字就会更白一些。

Let's look at the weights learned for all 10 of the output neurons. As suspected, they all look like somewhat blurry versions of our ten digits. They appear almost as though we averaged many images belonging to each digit class.

让我们来看看所有 10 个输出神经元经过学习后的权重。一如所料，他们看起来都很像模糊过的数字。它们看起来就像我们把每个数字的所有照片取了个平均。

{% include figure.html path="/images/figures/rolled_weights_mnist.png" caption="Visualizing the weights for all the output neurons of an MNIST classifier" %}

Suppose we receive an input from an image of a 2. **We can anticipate that the neuron responsible for classifying 2s should have a high value because its weights are such that high weights tend to align with pixels tending to be high in 2s. ？**For other neurons, _some_ of the weights will also line up with high-valued pixels, making their scores somewhat higher as well. However, there is much less overlap, and many of the high-valued pixels in those images will be negated by low weights in the 2 neuron. The activation function does not change this, because it is monotonic with respect to the input, that is, the higher the input, the higher the output.

假设我们输入一个 2 的图像。**可以预期，负责分类 2 的神经元会有一个比较高的值，因为这个图像会倾向于和 2 的像素点重合，因此负责 2 的神经元会有较高的权重。**对于其他神经元，**某些**权重也会因为和相关像素重叠，而变得有点高。然而，这种重叠还是比较少的，而且很多值很高的像素会被负责 2 的神经元中的低权重抵消掉。激活函数不会对这个有所改变，因为它对于输入值是单调的，也就是说，输入值越高，输出值越高。

We can interpret these weights as forming templates of the output classes. This is really fascinating because we never _told_ our network anything in advance about what these digits are or what they mean, yet they came to resemble those object classes anyway. This is a hint of what's really special about the internals of neural networks: they form _representations_ of the objects they are trained on, and it turns out these representations can be useful for much more than simple classification or prediction. We will take this representational capacity to the next level when we begin to study [convolutional neural networks](/ml4a/convnets/) but let's not get ahead of ourselves yet...

我们可以这样理解这些权重：它们是我们输出的那些 class（类）的样本。这是一件很赞的事，因为我们从来没有提前**告知**我们的神经网络任何事，比如这些数字是什么，或者它们代表什么意思，但是它却在模仿我们的目标 class。这暗示了我们这些神经网络真正特别的地方：它们构建了训练集的 _representations_（表示/表征），并且这些 representations 远不止用于分类和预测那么简单。当我们开始学习[卷积神经网络（convolutional neural networks）](/ml4a/convnets/)时，我们可以更进一步运用这种表示能力，但暂且先按下不表。

This raises many more questions than it provides answers, such as what happens to the weights when we add hidden layers? As we will soon see, the answer to this will build upon what we saw in the previous section in an intuitive way. But before we get to that, it will help to examine the performance of our neural network, and in particular, consider what sorts of mistakes it tends to make.

比起给出答案，它其实带给我们更多问题。比如说，隐藏层的权重怎么样了？当然很快可以发现，我们可以基于之前的部分直观地得到解答。但是在这之前，它会帮助我们检验神经网络的表现如何，尤其是考虑它可能会犯什么错。

## 0op5, 1 d14 17 2ga1n

Occasionally, our network will make mistakes that we can sympathize with. To my eye, it's not obvious that the first digit below is 9. One could easily mistake it for a 4, as our network did. Similarly, one could understand why the second digit, a 3, was misclassified by the network as an 8. The mistakes on the third and fourth digits below are more glaring. Almost any person would immediately recognize them as a 3 and a 2, respectively, yet our machine misinterpreted the first as a 5, and is nearly clueless on the second.

偶尔，我们的神经网络会犯一些可以理解的错误。就算是我来看，下方第一个数字也不明显像是 9。也许有人会把它当成 4，就像我们的神经网络一样。同样地，人们也会理解为什么第二个数字 3 会被神经网络当做 8。但是神经网络在第三个和第四个数字上犯的错就比较令人不解了。几乎任何人都可以立刻辨认出这是 3 和 2，然而我们的机器把第三个数字解读为 5，然后几乎无法分辨第四个是几。

{% include figure.html path="/images/figures/mnist-mistakes.png" caption="A selection of mistakes by our 1-layer MNIST network. The two on the left are understandable; the two on the right are more obvious errors." %}

Let's look more closely at the performance of the last neural network of the previous chapter, which achieved 90% accuracy on MNIST digits. One way we can do this is by looking at a confusion matrix, a table which breaks down our predictions into a table. In the following confusion matrix, the 10 rows correspond to the actual labels of the MNIST dataset, and the columns represent the predicted labels. For example, the cell at the 4th row and 6th column shows us that there were 71 instances in which an actual 3 was mislabeled by our neural network as a 5. The green diagonal of our confusion matrix shows us the quantities of correct predictions, whereas every other cell shows mistakes.

让我们来仔细看看上一章的最后一个神经网络的表现如何。这个神经网络在分类 MNIST 上达到了 90% 的精确度。我们可以用 confusion matrix（混淆矩阵）来检验它的性能。Confusion matrix 会把预测值以表格的形式表现出来。下面的这个混淆矩阵中，表格的 10 行分别对应 MNIST 数据集里的真实数字大小，10 列则分别代表预测值。举例来说，第四行第六列代表总共有 71 个样本，其中 3 个被神经网络误认为是 5。混淆矩阵中绿色的对角线显示了正确预测的数量，其他格子则表示错误预测。

Hover your mouse over each cell to get a sampling of the top instances from each cell, ordered by the network's confidence (probability) for the prediction.

把鼠标指向相应格子，可以看到对应的样本数据，按照置信度（confidence）排序。

{% include todo.html note="add description to confusion matrices" %}

{% include demo_insert.html path="/demos/confusion_mnist/" parent_div="post" %}

{% include todo.html note="fix overflow in right table" %}

We can also get some nice insights by plotting the top sample for each cell of the confusion matrix, as seen below.

我们也可以把混淆矩阵每个格子对应的第一个样本标在上面，如下图所示。这给我们带来了一个不错的视角。

{% include figure.html path="/images/figures/mnist-confusion-samples.png" caption="Top-confidence samples from an MNIST confusion matrix" %}

This gives us an impression of how the network learns to make certain kinds of predictions. Looking at first two columns, we see that our network appears to be looking for big loops to predict 0s, and thin lines to predict 1s, mistaking other digits if they happen to have those features.

这样我们可以直观看到神经网络是如何学习某种预测的。看前两列，可以发现在预测 0 时，神经网络试图寻找大的圆圈。在预测 1 时，会把有细长竖条形状的数字当做 1。


## Breaking our neural network 打破原有的神经网络

So far we've looked only at neural networks trained to identify handwritten digits. This gives us many insights but is a very easy choice of dataset, giving us many advantages; We have only ten classes, which are very well-defined and have relatively little internal variance among them. In most real-world scenarios, we are trying to classify images under much less ideal circumstances. Let's look at the performance of the same neural network on another dataset, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), a labeled set of 60,000 32x32 color images belonging to ten classes: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The following is a random sample of images from CIFAR-10.

到目前为止我们一直在讨论识别手写数字的神经网络。尽管它带给我们很多新的洞见，这个数据集其实非常简单；我们只有 10 个 class，而且每个都有清晰的定义，互相之间没有太多分歧。但在大多数现实世界的场景都没有这么完美的。我们可以看看同样的神经网络用来分类另一个数据集 CIFAR-10 的表现。[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 包含 60,000 张 32x32 的图片，也分 10 个 class：飞机，汽车，鸟，猫，鹿，狗，青蛙，马，船舶，和卡车。下面是 CIFAR-10 的一个随机样本。

{% include figure.html path="/images/figures/cifar-grid.png" caption="A random sample from CIFAR-10 image set" %}

{% include todo.html note="new demo, refresh for new random sample" %}

Right away, it's clear we must contend with the fact that these image classes differ in ways that we haven't dealt with yet. For example, cats can be facing different directions, have different colors and fur patterns, be **outscretched？** or curled up, and many other variations we don't encounter with handwritten digits. Photos of cats will also be cluttered with other objects, further complicating the problem. 

立刻我们就意识到，我们将面临很多新的问题，这些 class 和之前的非常不同。比如，猫可能是朝各种方向的，有不同的花色，有舒展的或者蜷缩着的，还有其他在处理手写数字时没遇到的多样性。有些猫的照片还会被其他物体遮挡，使这个问题更加复杂。

Sure enough, if we train a 2-layer neural network on these images, our accuracy reaches only 37%. That's still much better than taking random guesses (which would get us a 10% accuracy) but it's far short of the 90% our MNIST classifier achieves. When we start convolutional neural networks, we'll improve greatly on those numbers, for both MNIST and CIFAR-10. For now, we can get a more precise sense about the shortcomings of ordinary neural networks by inspecting their weights.

不出所料，如果我们训练一个 2 层的神经网络来分辨这些图像，准确度只有 37%。虽然这已经比随机猜测的准确度（10%）要高了，但还远远比不上 MNIST 的 90%。如果我们用卷积神经网络，不管是对于 MNIST 还是 CIFAR-10，预测准确度都会大大提高。至于现在，我们可以进一步了解普通神经网络的缺点。不妨先来仔细看看它的权重。

Let's repeat the earlier experiment of observing the weights of a 1-layer neural network with no hidden layer, except this time training on images from CIFAR-10. The weights appear below.

让我们重复之前的实验。还记得上一次如何观察 1 层（无隐藏层）的神经网络的权重吗？唯一的区别在于这次是训练 CIFAR-10 的图片。权重如下图所示。

{% include figure.html path="/images/figures/rolled_weights_cifar.png" caption="Visualizing the weights for 1-layer CIFAR-10 classifier" %}

Compared to the MNIST weights, these have fewer obvious features and far less definition to them. Certain details do make intuitive sense, e.g. airplanes and ships are mostly blue on the outer edges of the images, reflecting the tendency for those images to have blue skies or waters around them. Because the weights image for a particular class does correlate to an average of images belonging to that class, we can expect blobby average colors to come out, as before. But because the CIFAR classes are much less internally consistent, the well-defined "templates" we saw with MNIST are far less evident.

对比 MNIST 的权重可视化图，这些图看起来没有明显的特征，也更难定义。部分细节还是符合直觉的：飞机和轮船的外沿大都是蓝色的，意味着它们被蓝色的天空或者水包围着。因为将每个 class 的权重可视化处理之后，所得图片是与所有图片平均后的结果相关的，因此和之前一样，这些斑驳的色块反映了一定的平均结果。但是因为 CIFAR 数据集的内部一致性更低，所以这些“样本”远不如之前清晰明显。

Let's take a look at the confusion matrix associated with this CIFAR-10 classifier.

让我们来看看 CIFAR-10 的混淆矩阵。

{% include demo_insert.html path="/demos/confusion_cifar/" parent_div="post" %}

Not surprisingly, its performance is very poor, reaching only 37% accuracy. Clearly, our simple 1-layer neural network is not capable of capturing the complexity of this dataset. One way we can improve its performance somewhat is by introducing a hidden layer. The next section will analyze the effects of doing that.

并不意外，它的表现不佳，只有大约 37% 的准确度。很显然，只有 1 层的神经网络并不能胜任这个数据集的复杂度。想要提升它的表现，一种办法是引入隐藏层。接下来我们将会分析这样做的结果。

## Adding hidden layers 添加隐藏层

<!--
Hidden layers are essential here. One obvious way they can help is best exemplified by the weight image for the horse class. The vague template of a horse is discernible, but it appears as though there is a head on each side of the horse. Evidently, the horses in CIFAR-10 seem to be usually facing one way or the other. If we create a hidden layer, a horse classifier could benefit by allowing the network to learn a "right-facing horse" or a "left-facing horse" inside the intermediate layer -->

So far, we've focused on 1-layer neural networks where the inputs connect directly to the outputs. How do hidden layers affect our neural network? To see, let's try inserting a middle layer of ten neurons into our MNIST network. So now, our neural network for classifying handwritten digits looks like the following.

目前为止，我们主要集中在单层神经网络上，始终是输入层直接连到输出层。隐藏层会如何影响神经网络？为了探寻这个答案，我们不妨加入一个有 10 个神经元的中间层到我们的 MNIST 神经网络中。所以现在我们的神经网络看起来是这样：

{% include figure.html path="/images/figures/mnist_2layers.png" caption="2-layer neural network for MNIST" %}

**Our simple template metaphor in the 1-layer network above doesn't apply to this case, because we no longer have the 784 input pixels connecting directly to the output classes.？** In some sense, you could say that we had "forced" our original 1-layer network to learn those templates because each of the weights connected directly into a single class label, and thus only affected that class. But in the more complicated network that we have introduced now, the weights in the hidden layer affect _all ten_ of the neurons in the output layer. So how should we expect those weights to look now?

我们在单层神经网络中的“模板”（template）的比喻在这种情况下就不成立了，因为不再是 784 个输入的像素直接连到输出的 class 上。从某种程度上，可以说我们之前是“强迫”单层神经网络学习这些模板的，因为每个权重都是直接联系到某个 class 上，因此只影响那一个 class。但是在这个更复杂的神经网络中，隐藏层的权重会影响到**全部**输出层的神经元。所以，现在我们该如何看待这些权重？

To understand what's going on, we will visualize the weights in the first layer, as before, but we'll also look carefully at how their activations are then combined in the second layer to obtain class scores. **Recall that an image will generate a high activation in a particular neuron in the first layer if the image is largely sympathetic to that filter.** So the ten neurons in the hidden layer reflect the presence of those ten features in the original image. In the output layer, a single neuron, corresponding to a class label, is a weighted combination of those previous ten hidden activations. Let's look at them below.

像往常一样，为了更好地理解发生的事情，我们会将第一层的权重可视化，但是我们还将仔细分析激活它们会如何影响第二层。回忆一下，如果一幅图像和某个 filter（过滤器？）很匹配的话，它就会在第一层某个特定的神经元中得到一个较高的值。所以隐藏层的 10 个神经元反映了原始图像的 10 种特征。在输出层，对应一个 class 的神经元，是之前隐藏层的 10 个特征的加权平均。

{% include demo_insert.html path="/demos/f_mnist_weights/" parent_div="post" %}

Let's start with the first layer weights, visualized at the top. They don't look like the image class templates anymore, but rather more unfamiliar. Some look like pseudo-digits, and others appear to be components of digits: half loops, diagonal lines, holes, and so on.

让我们从第一层的权重开始看，顶端是它们的可视化结果。它们看起来不再像是“模板”，而更陌生。有些看起来像是假的数字，另一些看起来像是数字的某些部分：半个环，斜线，孔洞，等等。

The rows below the filter images correspond to our output neurons, one for each image class. The bars signify the weights associated to each of the ten filters' activations from the hidden layer. For example, the `0` class appears to favor first layer filters which are high along the outer rim (where a zero digit tends to appear). It **disfavors？** filters where pixels in the middle are low (where the hole in zeros is usually found). The `1` class is almost the opposite of this, preferring filters which are strong in the middle, where you might expect the vertical stroke of a `1` to be drawn.

Filter 图片下面的每行分别对应每个输出神经元。每条黑色柱体都代表了相应的权重。比如，‘0’ 类就比较认同第一层 filter，这个 filter 倾向于在外圈取得较大值（就像数字 0 那样）.它就**不？**很认同在中间取得较低值的 filter（0 中间的洞的位置）。‘1’类恰恰相反，倾向于认同中间很明显的 filter，正好是竖着一笔 1 的位置。

The advantage of this approach is flexibility. For each class, there is a wider array of input patterns that stimulate the corresponding output neuron. Each class can be triggered by the presence of several abstract features from the previous hidden layer, or some combination of them. Essentially, we can learn different kinds of zeros, different kinds of ones, and so on for each class. This will usually--but not always--improve the performance of the network for most tasks.

这种方法的优点是它的灵活性。对于每个 class，都有很多种输入的模式和输出的神经元相对应。每个 class 都可以被各种抽象的特征，或者几种特征的组合所触发。总体来说，我们可以学习不同种类的 0，不同的 1 等等。这通常会——虽然不总是会——提升神经网络处理任务的能力。

## Features and representations 特征和表示

Let's generalize some of what we've learned in this chapter. In single-layer and multi-layer neural networks, each layer has a similar function; it transforms data from the previous layer into a "higher-level" representation of that data. By "higher-level," we mean that it contains a compact and more salient representation of that data, in the way that a summary is a "high-level" representation of a book. For example, in the 2-layer network above, we mapped the "low-level" pixels into "higher-level" features found in digits (strokes, loops, etc) in the first layer, and then mapped those high-level features into an even higher-level representation in the next layer, that of the actual digits. This notion of transforming data into smaller but more meaningful information is at the heart of machine learning, and a primary capability of neural networks.

让我们来总结一下本章所学的知识。在单层和多层神经网络中，每层都有着相似的功能；它们将前一层的数据转化成“高级”的数据表征（representation）。“高级”指数据更加简洁且特征明显，就像摘要就是一本书的“高级”的表征。比如，在上面的 2 层的神经网络中，在第一层，我们把“更低等级”的像素映射到数字的“高级”的特征（笔触、环形等），然后在下一层把这些高级特征映射到更高级的特征——真实数字。把数据（data）转化为更小而精的信息（information），就是机器学习的核心，也是神经网络最主要的能力。

By adding a hidden layer into a neural network, we give it a chance to learn features at multiple levels of abstraction. This gives us a rich representation of the data, in which we have low-level features in the early layers, and high-level features in the later layers which are composed of the previous layers' features. 

通过给神经网络添加一层隐藏层（hidden layer），它就能学习不同程度的抽象特征。这使我们对数据有了丰富的表征方式。现在我们就有了前面神经层的低级特征，和后面神经层的高级特征，每个高级特征又由前层的多个低级特征组成。

As we saw, hidden layers can improve accuracy, but only to a limited extent. Adding more and more layers stops improving accuracy quickly, and comes at a computational cost -- we can't simply ask our neural network to memorize every possible version of an image class through its hidden layers. It turns out there is a better way, using [convolutional neural networks](/ml4a/convnets), which will be covered in a later chapter. 

如我们所见，隐藏层可以提高精确度，但是只能在一定程度上提高它。继续添加神经层不能更快地提高精确度，且会带来更大的计算成本——我们不能简单地要求神经网络去记住一张图片在隐藏层的每种可能性。令人惊喜的是，我们有更好的办法。下一章我们将讲解[卷积神经网络（convolutional neural networks）](/ml4a/convnets)。

## Further reading 推荐阅读

{% include todo.html note="summary / further reading" %}

<!--

https://cs231n.github.io/understanding-cnn/
http://cs.nyu.edu/~fergus/drafts/utexas2.pdf
http://arxiv.org/pdf/1312.6034v2.pdf
http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf
deepvis
mtyka recent stylenet
http://arxiv.org/pdf/1602.03616v1.pdf


2 layer softmax
now we see combination of higher level parts

deepvis, looks for text even though we didnt ask it


Tinker With a Neural Network Right Here in Your Browser. (viegas, wattenberg)
http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.62418&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
-->
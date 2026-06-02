# Literature Review: Vision Transformers for Emotion Recognition in Abstract Art
## COMP6001 - Computer Vision and Multimodal Machine Learning | Adelaide University


## 1. History of the Vision Transformer

The Vision Transformer (ViT) was introduced by Dosovitskiy et al. in 2021 in the seminal paper "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (ICLR 2021). ViT represented a fundamental departure from the convolutional paradigm that had dominated computer vision since the success of AlexNet (Krizhevsky et al., 2012).

The transformer architecture itself originated in natural language processing (NLP) with Vaswani et al.'s "Attention is All You Need" (NeurIPS 2017), which introduced the multi-head self-attention mechanism. Prior to ViT, attempts to apply self-attention to images were limited, models such as Non-Local Means (Wang et al., 2018) and Stand-Alone Self-Attention (Ramachandran et al., 2019) combined attention with convolutions rather than replacing them entirely.

ViT's key insight was to treat an image as a sequence of fixed-size patches (16×16 pixels), linearly embed each patch into a token, and process the resulting sequence with a standard Transformer encoder. A prepended [CLS] token aggregates global context and is passed to a classification head. This approach required large-scale pretraining (JFT-300M or ImageNet-21k) to outperform CNNs, as Transformers lack the inductive biases of convolutions (locality, translation equivariance).

---

## 2. Novelty of ViT Compared to Prior Works

ViT introduced several key novelties over prior computer vision methods:

### 2.1 Pure Self-Attention for Vision
Unlike prior hybrid models (e.g. ResNet + attention), ViT was the first architecture to apply a pure Transformer encoder directly to image patches with no convolutional layers. This demonstrated that self-attention alone is sufficient for competitive image recognition.

### 2.2 Global Receptive Field from Layer One
CNNs build up receptive fields hierarchically through stacked local filters. ViT's self-attention mechanism attends to all patches simultaneously from the very first layer, capturing long-range dependencies that CNNs can only access in deeper layers. This is particularly relevant for abstract art, where emotional content is conveyed through global compositional features, colour mood, spatial tension, and textural atmosphere across the full canvas, rather than local objects.

### 2.3 Learnable Positional Encodings
Rather than fixed sinusoidal encodings (as in the original Transformer), ViT uses learnable 1D positional embeddings that adapt to the specific spatial structure of the training data.

### 2.4 Scalability
ViT demonstrated superior scaling properties compared to CNNs. As model size and pretraining data increase, ViT's performance improves more steeply than ResNet, making it the foundation for subsequent large-scale vision models.

---

## 3. State-of-the-Art Methods

Following ViT, a rich ecosystem of transformer-based vision models has emerged:

### 3.1 DeiT (Data-efficient Image Transformers)
Touvron et al. (2021) introduced DeiT, which trains ViT-scale models on ImageNet alone (without JFT-300M) using knowledge distillation from a CNN teacher. DeiT introduced a distillation token alongside the CLS token, making ViT practical without massive datasets.

### 3.2 Swin Transformer
Liu et al. (2021) proposed the Swin Transformer, which introduces hierarchical feature maps and shifted window attention. By restricting self-attention to local windows and shifting them between layers, Swin achieves linear computational complexity with respect to image size, making it suitable for dense prediction tasks (detection, segmentation). Swin currently holds state-of-the-art results on many vision benchmarks.

### 3.3 BEiT and MAE
Bao et al. (2022) introduced BEiT (BERT Pre-training of Image Transformers), applying masked image modelling to ViT pretraining. He et al. (2022) proposed Masked Autoencoders (MAE), which mask 75% of image patches and reconstruct them, enabling highly efficient self-supervised pretraining of ViT models.

### 3.4 Emotion Recognition from Art
The ArtEmis dataset (Achlioptas et al., 2021) established the first large-scale benchmark for affective language grounded in visual art. Their image-based emotion classification baseline achieved 60.0% accuracy using a fine-tuned ResNet-32 on the dominant-emotion subset. This project extends that baseline by evaluating a Vision Transformer architecture on the same task, representing a novel application of self-attention mechanisms to art emotion recognition.

---

## 4. Research Gap

Despite significant advances in both Vision Transformers and affective computing, several research gaps remain in the specific domain of emotion recognition from abstract art:

### 4.1 ViT Applied to Abstract Art Emotion
The ArtEmis dataset baseline (Achlioptas et al., 2021) achieves 60.0% accuracy using a fine-tuned ResNet-32 on the dominant-emotion subset (Section 6, Achlioptas et al., 2021). No published work has systematically evaluated ViT architectures on this dataset, leaving open the question of whether global self-attention provides advantages over local CNN features for abstract emotional content.

### 4.2 Training ViT from Scratch on Small Datasets
Most ViT research assumes large-scale pretraining on ImageNet-21k or JFT-300M (Dosovitskiy et al., 2021). Training ViT from scratch on domain-specific small datasets such as the 15k subset used here remains challenging due to the absence of CNN-style inductive biases. Techniques such as DeiT-style distillation or MAE pretraining have not been evaluated on art emotion datasets.

### 4.3 Multimodal Fusion
The ArtEmis dataset is inherently multimodal, each painting is paired with a free-text emotional rationale. Achlioptas et al. (2021) train image and text emotion classifiers separately (Cemotion|image and Cemotion|text, Section 4.1), rather than fusing them. Effective fusion of visual and linguistic signals for art emotion recognition remains an open research problem.

### 4.4 Subjective Label Noise
Abstract art emotion annotation is inherently subjective. Achlioptas et al. (2021) report that 61% of all annotated artworks received at least one positive and one negative emotional reaction simultaneously (Section 3.2), indicating significant inter-annotator disagreement. Robust training under subjective label noise in art emotion recognition has not been systematically studied.

---

## 5. Potential Drawbacks of ViT

### 5.1 Data Hungry
ViT requires large amounts of training data to outperform CNNs. Dosovitskiy et al. (2021) explicitly note that ViT trained on mid-sized datasets without strong regularisation yields modest accuracies compared to ResNets, as the model lacks the inductive biases of convolutions, locality and translation equivariance, that CNNs encode by design.

### 5.2 Computational Cost
ViT's self-attention has quadratic complexity O(n²) with respect to the number of patches (Vaswani et al., 2017). For a 224×224 image with 16×16 patches this yields 196 patches, manageable but more expensive than CNN inference at the same resolution.

### 5.3 Lack of Inductive Bias
Unlike CNNs which encode locality and translation equivariance by design, ViT must learn these properties entirely from data (Dosovitskiy et al., 2021). This makes ViT less sample-efficient and harder to train from scratch on small domain-specific datasets such as the 15k art emotion subset used in this project.

### 5.4 Positional Encoding Limitations
ViT uses learnable 1D positional embeddings added to patch tokens (Dosovitskiy et al., 2021). When inference is performed at a different resolution than training, these embeddings must be interpolated, which can degrade performance. This limits flexibility compared to CNN architectures which are resolution-agnostic by design.

### 5.5 Interpretability
While attention rollout (Abnar & Zuidema, 2020) provides some interpretability by aggregating attention weights across layers, ViT attention maps are less straightforward to interpret than CNN-based methods such as Grad-CAM (Selvaraju et al., 2017), as attention is distributed across all patches and heads simultaneously.


## References

- Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. ACL 2020.
- Achlioptas, P., Ovsjanikov, M., Haydarov, K., Elhoseiny, M., & Guibas, L. (2021). ArtEmis: Affective Language for Visual Art. CVPR 2021.
- Bao, H., Dong, L., & Wei, F. (2022). BEiT: BERT Pre-Training of Image Transformers. ICLR 2022.
- Dosovitskiy, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR 2022.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS 2012.
- Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV 2021.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.
- Touvron, H., et al. (2021). Training Data-Efficient Image Transformers & Distillation Through Attention. ICML 2021.
- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS 2017.
- Wang, X., et al. (2018). Non-local Neural Networks. CVPR 2018.
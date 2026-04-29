# Fine-tuning Seq2Seq Language Models for Topic Labelling

## Problem Statement

Topic modeling models are widely used in a plephora of applications however they output a list of topic words which is usually not desirable by stakeholders and requires a further step which is topic labelling. Common practices include using LLM APIs such as ChatGPT, which is uncontrollable and highly stochastic due to the wider, unneeded knowledge that the language models have, and manual labelling which is not practical due to the time needed and subjectivity. For this, we suggest fine-tuning seq2seq models, namely T5-base and T5-large, for the topic labelling task.

## Experiments and Surveys

### Comparing LLMs Performance in Topic Labelling

In our cluster's previous work we had collected 67 topics, each composed of 8 topic words each in addition to topic_labels which were assigned by ChatGPT API and verified by our supervisors since they are domain experts in the topics' research fields.

For each set of topic words we generated an appropriate topic label using the exact same prompt using: GPT5, Claude, Perplexity, Gemini Pro, Grok, and DeepSeek. Then, we embedded the topic_words and all 6 generated labels using allenai-specter model since it was trained on scientific data simillar to our use case. Then cosine similarity was measured between the embeddings of the set of topic-words and each generated label's embeddings. Models were compared based on mean cosine similarity with the following results: {Grok: 0.887, Gemini: 0.885, Perplexity: 0.883, Claude: 0.881, GPT5: 0.878, DeepSeek: 0.873}. Grok and Gemini were superior to other models, however we decided to proceed with Gemini due to it's ease in integration with Google Sheets (AI-generated columns use Gemini).

### Verifying the Dataset

Our hypothesis is: LLMs are capable of generating high quality topic_words - topic_label pairs for a given research field.

To conduct this experiment, we requested the assistance of 35 PhD holders from The German University in Cairo from across 8 faculties, namely: Media Engineering and Technology, Information Engineering and Technology, Engineering and Materials Science, Pharmacy and Biotechnology, Management Technology, Applied Sciences and Arts, in addition to the Mathematics and Physics depratments. For each professor, 3-6 research fields are determined through their online research profiles (e.g. Google Scholar). For every unique research field, Gemini was used to generate 7 topic_words-topic_labels pairs using the following prompt:

```
You are an expert research scientist, with a deep focus on [Your Research Field]. Your task is to generate 7 unique (topic_words, topic_label) pairs relevant to this field. Follow these strict guidelines: 1. `topic_words`: This must be a JSON list of 5 lowercase, single-word strings. These words should be highly related keywords, as if they were the top 5 tokens from an LDA topic model. 2. `topic_label`: This must be a concise, human-readable string (2-5 words) that accurately summarizes the `topic_words`. Here are examples from a *different* field (Computational Biology) to show the exact format: [ { "topic_words": ["gene", "expression", "rna-seq", "transcriptome", "differential"], "topic_label": "Gene Expression Analysis" }, { "topic_words": ["protein", "structure", "folding", "docking", "molecular"], "topic_label": "Protein Structure & Docking" }, { "topic_words": ["crispr", "cas9", "editing", "genome", "mutation"], "topic_label": "CRISPR Genome Editing" } ] Now, generate 7 new pairs specifically for my field: * **Sub-discipline:** [Your Sub-discipline] * **Research Field:** [Your Research Field] Produce the output as a single, valid JSON list of objects.
```

Each professor was asked to vote on each pair of his assigned research fields wether he agrees or disagrees. The survey was held in a web based portal with clear task instructions and examples. Traps were embedded within the survey to detect professors who had voted `Agree` on all pairs without examining them thoroughly, all their votes will be disregarded in the final results. Out of 931 pairs, 812 pairs received an `Agree` vote achieving a 87.22% success rate. We test our hypothesis using One-Sample Proportion Z-test to see if this result is significantly better than a random guess (50% since it is a binary vote).

- Sample Proportion (p hat): 0.8722
- Hypothesized Proportion (p node): 0.50 (the "null" baseline)
- Sample Size (n): 931

Z = (p hat - p node)/ sqrt((p node - (1- p node))/ sample size) = 22.71

- Z-score: 22.71
- p-value: < 0.00001

Since the p-value is essentially zero (well below the standard threshold of 0.05), we can confidently conclude that the experts' agreement is statistically significant and the results are not due to random chance.

The conclusion we gain from this hypothesis is that we can confidently generate topic_words-topic-label pairs for a specific research field.

## Datasets

### Verified Dataset

The topic_words-topic_label pairs from the previous survey were filtered from pairs which received a disagree vote. Trapped participants' responses were filtered as well. Forming 812 verified pairs to be reffered to later as the Verified Dataset.

### Taxonomy Dataset Generation

Using Elsevier's Digital Commons Three-Tiered Taxonomy of Academic Disciplines, we generated for each third-level academic discipline (e.g. Physical Sciences and Mathematics: Computer Sciences: Software Engineering) a list of topic_words-topic_label pairs using the exact same prompt used earlier. The taxanomoy includes 913 third-level research fields, however 0nly 911 were used since the other 2 were denied from a response by Gemini due to ethical issues (e.g. Holocaust and Genocide Studies). Resulting into 6377 of topic_words-topic_label pairs across all scientific research fields to be referred to later as the Taxonomy Dataset.

## Fine-tuning

To automatically generate concise and semantically meaningful topic labels, we fine-tuned a pretrained Text-to-Text Transfer Transformer (T5) model for conditional text generation. The task is formulated as a sequence-to-sequence problem, where a set of topic-representative keywords serves as input and a short descriptive topic label is generated as output.

### Model Architecture and Initialization

We adopt a pretrained T5 encoder–decoder architecture, which is well-suited for text generation tasks due to its unified text-to-text formulation. Given an input sequence consisting of topic keywords embedded within a natural language prompt, the encoder produces contextual representations that are decoded into a concise topic label.

The model is initialized from a pretrained checkpoint (`model_name`) which is either T5-base or T5-large. T5-small was eliminated due to poor capabilities while larger T5 models were eliminated due to theoretically not being affected enough by our datasets' small sizes., Using a pretrained model allows us to leverage rich semantic knowledge acquired during large-scale pretraining. This initialization significantly reduces the amount of task-specific data required and improves generalization.

### Encoder Freezing Strategy

To preserve the pretrained semantic representations while allowing task-specific adaptation, we employ partial encoder freezing. Specifically, the first four encoder layers are frozen during training. Lower encoder layers in transformer architectures typically capture general linguistic and semantic features, whereas higher layers encode more task-specific abstractions. By freezing the initial layers, we prevent catastrophic forgetting of fundamental semantic knowledge while allowing the upper encoder layers and the decoder to specialize in topic label generation. This strategy is particularly effective when training data is limited or noisy.

### Optimization and Regularization

The model is optimized using the AdamW optimizer with decoupled weight decay. The learning rate is deliberately set to a small value = 5e-5 . This low learning rate ensures stable fine-tuning and minimizes destructive updates to pretrained weights. Weight decay is applied with a coefficient of 0.01 to regularize the model and reduce overfitting, particularly in the decoder layers that are fully trainable.

Additionally, gradient clipping with a maximum norm of 1.0 is applied during training to prevent exploding gradients and improve optimization stability.

### Training Procedure

The dataset is split into training and validation subsets, with a portion reserved for validation to monitor generalization performance. During training, data augmentation is applied exclusively to the training set to increase robustness, while the validation set remains unaltered.

A linear learning rate scheduler with warm-up is used, where the warm-up phase spans 10% of the total training steps. This gradual increase in learning rate at the beginning of training helps stabilize optimization when fine-tuning large pretrained models.

Early stopping is employed based on validation loss, with a patience threshold to prevent overfitting and unnecessary training epochs. The model checkpoint yielding the lowest validation loss is retained as the final model.

### Inference and Decoding Constraints

At inference time, topic labels are generated using beam search decoding with additional constraints to promote diversity and readability. Repetition penalties, n-gram blocking, and length constraints are applied to avoid trivial copying of input keywords and to encourage concise, human-interpretable labels.

Overall, this fine-tuning strategy balances semantic preservation, task adaptation, and training stability, resulting in a robust topic labeling model capable of producing high-quality descriptive labels from keyword-based topic representations.

Below is an **extended scientific subsection** that you can **append directly** to the previous section. It introduces a **clear mathematical formalization** of the sequence-to-sequence topic labeling task and aligns well with thesis / journal expectations.

---

### Mathematical Formulation of the Topic Labeling Task

Let a discovered topic be represented by an ordered set of representative keywords:

[
\mathbf{w} = (w_1, w_2, \dots, w_n),
]

where each ( w_i ) corresponds to a high-importance term extracted from a topic modeling algorithm. The objective is to generate a concise textual label ( \mathbf{y} ) that semantically summarizes the underlying topic:

[
\mathbf{y} = (y_1, y_2, \dots, y_m),
]

where ( \mathbf{y} ) is a short natural-language sequence describing the topic.

#### Input–Output Mapping

The task is formulated as a conditional sequence-to-sequence learning problem. A natural language prompt is constructed by embedding the topic keywords into a descriptive template:

[
\mathbf{x} = \text{Prompt}(w_1, \dots, w_n),
]

where ( \mathbf{x} ) is the input token sequence provided to the model. The model learns a conditional distribution over output sequences:

[
p_\theta(\mathbf{y} \mid \mathbf{x}),
]

parameterized by ( \theta ), the set of trainable parameters of the T5 model.

#### Encoder–Decoder Architecture

The encoder transforms the input sequence ( \mathbf{x} ) into a sequence of contextualized hidden states:

[
\mathbf{H} = \text{Encoder}_\theta(\mathbf{x}) = (h_1, h_2, \dots, h_T),
]

where ( T ) is the input sequence length and each ( h_t \in \mathbb{R}^d ) represents a contextual embedding.

The decoder generates the output label autoregressively. At decoding step ( t ), the probability of generating token ( y_t ) is given by:

[
p_\theta(y_t \mid y_{<t}, \mathbf{x}) = \text{Decoder}*\theta(y*{<t}, \mathbf{H}),
]

where ( y*{<t} = (y_1, \dots, y*{t-1}) ).

The joint probability of the output sequence is therefore:

[
p_\theta(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{m} p_\theta(y_t \mid y_{<t}, \mathbf{x}).
]

#### Training Objective

The model is trained to minimize the negative log-likelihood of the reference topic labels in the training dataset. Given a dataset ( \mathcal{D} = {(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})}\_{i=1}^{N} ), the training objective is defined as:

[
\mathcal{L}(\theta) = - \sum_{i=1}^{N} \sum_{t=1}^{m_i} \log p_\theta \left( y_t^{(i)} \mid y_{<t}^{(i)}, \mathbf{x}^{(i)} \right).
]

This objective corresponds to token-level cross-entropy loss and encourages the model to generate accurate and fluent topic labels conditioned on the input keywords.

#### Parameter Freezing Constraint

Let ( \theta = {\theta_e^{(1:L)}, \theta_d} ) denote the model parameters, where ( \theta_e^{(l)} ) represents the parameters of the ( l )-th encoder layer and ( \theta_d ) represents the decoder parameters. During training, the parameters of the first ( K = 4 ) encoder layers are frozen:

[
\frac{\partial \mathcal{L}}{\partial \theta_e^{(l)}} = 0 \quad \forall l \in {1, \dots, K}.
]

Only the remaining encoder layers ( \theta_e^{(K+1:L)} ) and the decoder parameters ( \theta_d ) are updated. This constraint preserves low-level semantic representations while enabling task-specific adaptation.

#### Inference

At inference time, the optimal topic label is obtained by decoding the most probable output sequence:

[
\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} ; p_\theta(\mathbf{y} \mid \mathbf{x}),
]

which is approximated using beam search with additional constraints to discourage repetition and enforce concise output.

## Results

Using each collected dataset we fine-tuned a T5-base model adn a T5-large model. Resulting in 4 different models: Verified Base, Verified Large, Taxonomy Base, and Taxonomy Large. A test set was prepared which is 161 pairs within the domain expertise of our supervisors. For each row, 4 labels were inferred using the 4 models stated. Assigned labels and generated labels were all embedded using allen-ai specter model. Cosine similarity was measured between all generated labels for performance evaluation. Performance: {Verified Base: {mean: 0.628; std: 0.201, min: 0.081; max: 1.00}; Verified Large: {mean: 0.644; std: 0.197, min: 0.085; max: 1.00}; Taxonomy Base: {mean: 0.627; std: 0.206, min: 0.031; max: 1.00}; Taxonomy Large: {mean: 0.645; std: 0.195, min: 0.017; max: 1.00}}. Many of the rows which had very low cosine similarities had abbreviations, for example "SHAP" and "LIME" which are Explainable AI methods were interpretted by the seq2seq model as a misspelled "shape" and "Lime" the fruit.

TODO

- Compare T5 outputs with Grok
- Find a different embedding model for keywords and short phrases
- Human evaluation of final output

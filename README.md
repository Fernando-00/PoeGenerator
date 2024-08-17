# PoeGenerator

- Using a modified pre-trained GPT-2 model we will fine tune it to generate text in the style of Edgar Allan Poe. To evaluate accuracy a pre-trained BERT model will be utilized to classify between Poe text vs Non-Poe.
- Data Source 1: the Poe Museum (https://poemuseum.org/poes-complete-work)
 Contains Poe’s poems and short story collections
Extracted 65 Poems and some short stories
- Data Source 2: Collection of all Shakespearean sonnets, organized by Martin Gorner. (https://github.com/martin-gorner/...)
Extracted 50 sonnets.

 **Classifier**
- Pre-trained BERT model
- Evaluating a collection of Shakespearean sonnets against poems by Poe
- Data is pre-processed, labeled, and stored in a pandas dataframe
- Model is trained with 3 epochs, 8 batch size
- Once model is trained, we load the dataset and evaluate it
- Input text to model, which classifies it as Poe or not Poe to assess generation quality

**Generator**
- GPT2 Transformer Layers: Standard transformer layers
- Adapters: Inserted after each transformer layer to adapt the model to the specific task of style transfer.
- Layer Normalization: Applied before generating the final output logits.
- Language Modeling Head: Computes the logits for language modeling
- Also calculates the loss if labels are provided.

  **Results**
- Classification:
- Training Accuracy: 96.65%
- Testing Accuracy: 96.14%
- Generation:
- Correct Rate: 92%

- Example: [How can i then return in happy plight,] “d — but to the night’s; and all my soul. from a more of her own is not be it was their heart! for this have power that— 




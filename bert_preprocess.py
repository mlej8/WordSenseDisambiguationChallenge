def preprocess_bert(sentences, max_len):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    number_of_sentences = len(sentences)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    for counter, sent in enumerate(sentences):
        print('======== Encoding sentence {} / {} ========'.format(counter + 1, number_of_sentences))
        """ 
        .encode_plus() will:
          (1) Tokenize the sentence.
          (2) Prepend the `[CLS]` token to the start.
          (3) Append the `[SEP]` token to the end.
          (4) Map tokens to their IDs.
          (5) Pad or truncate the sentence to `max_length`
          (6) Create attention masks for [PAD] tokens.
        """
        
        encoded_dict = tokenizer.encode_plus(
                            sent,                          # Sentence to encode.
                            add_special_tokens = True,     # Add '[CLS]' and '[SEP]'
                            max_length = max_len,          # Pad & truncate all sentences to max length
                            pad_to_max_length = True,
                            truncation=True,
                            return_attention_mask = True,  # Construct attn. masks.
                            return_tensors = 'pt',         # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids = encoded_dict['input_ids']
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks = encoded_dict['attention_mask']

    return input_ids, attention_masks

def get_predictions(sentences,labels):
    predictions = []
    for sent in sentences: 
        text = "[CLS] " + sent + " [SEP]"
        tokenized_text = tokenizer.tokenize(text) 
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) 
        masked_index = tokenized_text.index('[MASK]')

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        model.eval()

        # Predict all tokens
        with torch.no_grad():
            preds = model(tokens_tensor, segments_tensors)

        predicted_index = torch.argmax(preds[0][0][masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        predictions.append(predicted_token)

    predictions = [0 if "a" else 1 for prediction in predictions]
    accuracy = accuracy_score(labels, predictions)
    print("Predicted {} out of {} correctly\nAccuracy score: {:.2f}".format(accuracy * len(predictions), len(predictions), accuracy))
    return predictions


def whole_sentence_feature_extraction(sentences):
    """ Given a list of sentences in the form of lists of word tokens, return a feature vector and a target vector for each training instance """
    x = []
    y = []
    for word_tokens in sentences:
        for counter, token in enumerate(word_tokens):
            if token == "a": 
                words_before = word_tokens[:counter]
                words_after = word_tokens[counter+1:]
                sentence = " ".join(words_before) + " [MASK] " + " ".join(words_after) + "." 
                sentence = sentence.capitalize().replace("[mask]", "[MASK]")
                x.append(sentence)
                y.append(0) # 0 stands for "a"
            elif token == "the":      
                words_before = word_tokens[:counter]
                words_after = word_tokens[counter+1:]
                sentence = " ".join(words_before) + " [MASK] " + " ".join(words_after) + "." 
                sentence = sentence.capitalize().replace("[mask]", "[MASK]")
                x.append(sentence)
                y.append(1) # 1 stands for "the"
    return x, y

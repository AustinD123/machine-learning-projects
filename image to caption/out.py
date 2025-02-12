import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
from main_model import fullmodel
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the trained model
model_path = "checkpoints/model_epoch_5.pth"  # Update with your trained model's checkpoint path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fullmodel(embed_size=300, decoder_dim=512, encoder_dim=2048, attention_dim=256)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
model.to(device)

# Image Preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Caption Generation
def generate_caption(model, image_path, tokenizer, max_seq_length=20):
    image_tensor = preprocess_image(image_path).to(device)
    
    # Forward pass through the encoder
    with torch.no_grad():
        features = model.encoder(image_tensor)

    # Prepare the initial decoder input (start token)
    caption_input = torch.tensor([tokenizer.cls_token_id]).unsqueeze(0).to(device)

    hidden_state, cell_state = model.decoder.inithidden(features)
    caption_generated = []

    for _ in range(max_seq_length):
        # Get word embeddings and attention context
        embeds = model.decoder.embedding(caption_input)
        context, alpha = model.decoder.attention(features, hidden_state)

        # LSTM step
        lstm_input = torch.cat((embeds.squeeze(1), context), dim=1)
        hidden_state, cell_state = model.decoder.lstm_cell(lstm_input, (hidden_state, cell_state))

        # Output word probabilities
        output_logits = model.decoder.fcn(hidden_state)
        predicted_word_idx = torch.argmax(output_logits, dim=1).item()

        # Stop if reaching the [SEP] token
        if predicted_word_idx == tokenizer.sep_token_id:
            break

        # Append word to generated caption
        caption_generated.append(tokenizer.decode(predicted_word_idx))
        caption_input = torch.tensor([[predicted_word_idx]]).to(device)

    return ' '.join(caption_generated)

# Test the function
image_path = "test/1.jpg"  # Replace with your image path
caption = generate_caption(model, image_path, tokenizer)
print("Generated Caption:", caption)

from decoder import Decoder
import torch


vocab = {
    "<unk>": 0,    # Token ID 0 for unknown token
    "hello": 1,    # Token ID 1 for "hello"
    "world": 2,    # Token ID 2 for "world"
    "how": 3,      # Token ID 3 for "how"
    "are": 4,      # Token ID 4 for "are"
    "you": 5,      # Token ID 5 for "you"
    "<eos>": 6     # Token ID 6 for End of Sequence (EOS)
}
vocab2={v:k for k,v in vocab.items()}

input_output_pairs = [
    ([1, 2], [3, 4, 5, 6]),  # Input: "hello world" → Output: "how are you <eos>"
    ([3, 4], [5, 6]),        # Input: "how are" → Output: "you <eos>"
    ([2, 1], [4, 5, 6]),     # Input: "world hello" → Output: "are you <eos>"
    ([5, 3], [4, 2, 1, 6]),  # Input: "you how" → Output: "are world hello <eos>"
    ([4, 5], [3, 2, 6]),     # Input: "are you" → Output: "how world <eos>"
    ([1], [2, 6]),           # Input: "hello" → Output: "world <eos>"
    ([2], [1, 6])            # Input: "world" → Output: "hello <eos>"
]


model=Decoder(num_tokens=len(vocab),d_model=2,max_len=6)
model.load_state_dict(torch.load("trained_transformer.pth"))#loaded from training

def predict_token(input_tokens:torch.Tensor,max_length:int):
    
    token=""
    output=""
    for i in range(len(input_tokens),max_length):
        if token=="<eos>":
            break
        pred=model(input_tokens)
        token=vocab2[int(torch.argmax(pred[-1]))]
        output+=token
        input_tokens=torch.cat((input_tokens,torch.argmax(pred[-1]).unsqueeze(0)),0)
        
    return input_tokens


def predict_sentence(input_tokens:torch.Tensor,max_length:int):
    
    token=""
    output=""
    for i in range(len(input_tokens),max_length):
        if token=="<eos>":
            break
        pred=model(input_tokens)
        token=vocab2[int(torch.argmax(pred[-1]))]
        output+=token
        input_tokens=torch.cat((input_tokens,torch.argmax(pred[-1]).unsqueeze(0)),0)
    return output


model.eval()
#test with one output
print("""0:Input: hello world → Output: how are you <eos>
    1:Input: how are → Output: you <eos>
    2: Input: "world hello" → Output: "are you <eos>"
    3: Input: "you how" → Output: "are world hello <eos>"
    4: Input: "are you" → Output: "how world <eos>"
    5: Input: "hello" → Output: "world <eos>"          
    6: Input: "world" → Output: "hello <eos>""")
while True:
    inx=int(input("enter the input from the above:"))
    inp=input_output_pairs[inx][0]
    output=predict_sentence(input_tokens=torch.tensor(inp),max_length=6)
    print("transformer:",output)
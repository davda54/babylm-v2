import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_mlm(prompt, answer, tokenizer, model, device, verbose=False):

    prompt, ending = prompt.split("{answer}")

    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Ending: {ending}")

    inputs = tokenizer.encode(prompt.strip(), add_special_tokens=False).ids
    ending = tokenizer.encode(f'#{ending.strip()}', add_special_tokens=False).ids[1:]
    gold_output = torch.tensor([tokenizer.encode(answer, add_special_tokens=False).ids]).to(device)

    inputs = [tokenizer.token_to_id("␂")] + inputs + [tokenizer.token_to_id("␥")] * gold_output.size(1) + ending + [tokenizer.token_to_id("␃")]
    inputs = torch.tensor(inputs).unsqueeze(0).to(device)

    if verbose:
        print(f"Prompt: {inputs}")
        print(f"Ending: {ending}")
        print(f"Answer: {gold_output}")

    mask = torch.zeros_like(inputs, dtype=torch.bool).to(device)
    logits = model(inputs.t(), mask.unsqueeze(1)).transpose(0, 1)[0, -(gold_output.size(1) + len(ending) + 1):-(len(ending) + 1)]
    loss = F.cross_entropy(logits, gold_output[0])

    prediction = tokenizer.decode(logits.argmax(-1).tolist())

    if verbose:
        print(f"Prediction: {prediction}")
        print()
        if prediction.strip() != answer.strip():
            print(f"Wrong answer: {prediction} != {answer}")

    return prediction, loss


@torch.no_grad()
def evaluate_mlm_shift(prompt, answer, tokenizer, model, device, verbose=False):

    verbose = False

    prompt, ending = prompt.split("{answer}")

    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Ending: {ending}")

    inputs = tokenizer.encode(prompt.strip(), add_special_tokens=False).ids
    ending = tokenizer.encode(f'#{ending.strip()}', add_special_tokens=False).ids[1:]
    gold_output = torch.tensor([tokenizer.encode(answer, add_special_tokens=False).ids]).to(device)

    inputs = [tokenizer.token_to_id("␂")] + inputs + [tokenizer.token_to_id("␥")] * gold_output.size(1) + ending
    inputs = torch.tensor(inputs).unsqueeze(0).to(device)

    if verbose:
        print(f"Prompt: {inputs}")
        print(f"Ending: {ending}")
        print(f"Answer: {gold_output}")

    mask = torch.zeros_like(inputs).to(device)
    logits = model(inputs.t(), mask).transpose(0, 1)[0, -(gold_output.size(1) + len(ending) + 1):-(len(ending) + 1)]
    loss = F.cross_entropy(logits, gold_output[0])

    prediction = tokenizer.decode(logits.argmax(-1).tolist())

    if verbose:
        print(f"Prediction: {prediction}")
        print()
        if prediction.strip() != answer.strip():
            print(f"Wrong answer: {prediction} != {answer}")

    return prediction, loss


@torch.no_grad()
def evaluate_causal(prompt, answer, tokenizer, model, device, verbose=False):

    prompt, ending = prompt.split("{answer}")

    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Ending: {ending}")

    inputs = tokenizer.encode(prompt.strip(), add_special_tokens=False).ids
    ending = tokenizer.encode(f'#{ending.strip()}', add_special_tokens=False).ids[1:]
    gold_output = tokenizer.encode(answer, add_special_tokens=False).ids

    inputs = [tokenizer.token_to_id("␂")] + inputs + gold_output + ending
    inputs = torch.tensor(inputs).unsqueeze(0).to(device)

    gold_output = torch.tensor([gold_output]).to(device)

    if verbose:
        print(f"Prompt: {inputs}")
        print(f"Ending: {ending}")
        print(f"Answer: {gold_output}")

    with torch.no_grad():
        mask = torch.ones(inputs.size(1), inputs.size(1), dtype=torch.bool).triu(diagonal=1).unsqueeze(0).to(device)
        logits = model(inputs.t(), mask).transpose(0, 1)[0, -(gold_output.size(1) + len(ending) + 1):-(len(ending)+1)]
        first_loss = F.cross_entropy(logits[0], gold_output[0][0])
        loss = F.cross_entropy(logits, gold_output[0])

    prediction = tokenizer.decode(logits.argmax(-1).tolist())

    if verbose:
        print(f"Prediction: {prediction}")
        print()
        if prediction.strip() != answer.strip():
            print(f"Wrong answer: {prediction} != {answer}")

    return prediction, loss, first_loss


@torch.no_grad()
def evaluate_prefix(prompt, answer, tokenizer, model, device, verbose=False):

    verbose = False

    prompt, ending = prompt.split("{answer}")

    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Ending: {ending}")

    tokens = tokenizer.encode(prompt.strip(), add_special_tokens=False).ids
    ending = tokenizer.encode(f'#{ending.strip()}', add_special_tokens=False).ids[1:]
    gold_output = tokenizer.encode(answer, add_special_tokens=False).ids

    inputs = [tokenizer.token_to_id("␂")] + tokens + gold_output + ending
    inputs = torch.tensor(inputs).unsqueeze(0).to(device)

    gold_output = torch.tensor([gold_output]).to(device)

    if verbose:
        print(f"Prompt: {inputs}")
        print(f"Ending: {ending}")
        print(f"Answer: {gold_output}")

    mask = torch.ones(inputs.size(1), inputs.size(1), dtype=torch.bool).triu(diagonal=1).unsqueeze(0).to(device)
    mask[:, :len(tokens)+1, :len(tokens)+1] = False
    logits = model(inputs.t(), mask).transpose(0, 1)[0, -(gold_output.size(1) + len(ending) + 1):-(len(ending) + 1)]
    first_loss = F.cross_entropy(logits[0], gold_output[0][0])
    loss = F.cross_entropy(logits, gold_output[0])

    prediction = tokenizer.decode(logits.argmax(-1).tolist())

    if verbose:
        print(f"Prediction: {prediction}")
        print()
        if prediction.strip() != answer.strip():
            print(f"Wrong answer: {prediction} != {answer}")

    return prediction, loss, first_loss

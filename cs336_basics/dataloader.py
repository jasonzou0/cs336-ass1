import torch
import random

def get_batch(dataset: list, context_length: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        data (list): _description_
        batch_size (int): _description_
        context_length (int): _description_
        device (str): _description_

    """ 
    # Ensure enough data for at least one full sample
    if len(dataset) < context_length + 1:
        raise ValueError("Dataset too small for context_length + 1")

    max_start = len(dataset) - context_length - 1
    assert max_start >= 0, "Dataset too small for context_length"

    starts = []
    for _ in range(batch_size):
        start = random.randint(0, max_start)
        starts.append(start)

    samples = []
    targets = []
    for start in starts:
        sample = torch.tensor(dataset[start:start+context_length], device=device)
        target = torch.tensor(dataset[start+1:start+context_length+1], device=device)
        samples.append(sample)
        targets.append(target)

    output_sample = torch.stack(samples, dim=0)
    output_target = torch.stack(targets, dim=0)
    return (output_sample, output_target)


    
    
    
    

    # output_sample=torch.Tensor([]).to(device)
    # output_target=torch.Tensor([]).to(device)
    # # for i in range(0, len(dataset), context_length):
    # i=0
    # while i+1<len(dataset):
    #     sample=torch.tensor(dataset[i:i+context_length]).to(device)
    #     target=torch.tensor(dataset[i+1:i+context_length+1]).to(device)
    #     if len(sample)<context_length or len(target)<context_length:
    #         # restart from beginning if we reach the end
    #         i=0
    #         continue
    #     output_sample=torch.cat((output_sample, sample.unsqueeze(0)), dim=0)
    #     output_target=torch.cat((output_target, target.unsqueeze(0)), dim=0)
    #     if output_sample.shape[0]==batch_size:
    #         # print(f"Yielding batch of size {output_sample.shape}")
    #         # print(f"Sample: {output_sample}")
    #         # print(f"Target: {output_target}")
    #         # yield (output_sample, output_target)
    #         # output_sample=torch.Tensor([]).to(device)
    #         # output_target=torch.Tensor([]).to(device)
    #         return (output_sample, output_target)
        
    #     i=i+context_length

    # return (output_sample, output_target)
    # # if output_sample.shape[0]>0:
    # #     # yield (output_sample, output_target)
    # #     return (output_sample, output_target)
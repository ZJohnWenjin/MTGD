import torch

# model,x0,t,noise,beta
def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, condition,pet_type,keepdim=False,loss_moment=False):
    # alpha cumprod
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # get xt
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    # predict noise from xt and t
    output = model(x, t.float(),condition,pet_type)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    if loss_moment:
        return (e - output).square().mean(dim=list(range(1, len(output.shape))))
        # return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=1)
    else:
        # b 1 h w -> b
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}

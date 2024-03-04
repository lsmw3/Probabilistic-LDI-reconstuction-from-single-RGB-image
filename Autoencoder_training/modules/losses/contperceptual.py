import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", diff_loss_weight = 0.0, relaxation_factor = 0.0):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        if self.perceptual_weight >= -0.000001:
            self.perceptual_loss = LPIPS().eval()

        self.diff_loss_weight = diff_loss_weight
        self.relaxation_factor = relaxation_factor

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def calculate_diff_loss(self, pred_layer, pred_layer_2, gt_layer, gt_layer_2):
        # pred_layer: [B, H, W]
        # pred_layer_2: [B, H, W]
        # gt_layer: [B, H, W]
        # gt_layer_2: [B, H, W]

        gt_diff1 = gt_layer[:, 1:, :] - gt_layer_2[:, :-1, :]
        pred_diff1 = pred_layer[:, 1:, :] - pred_layer_2[:, :-1, :]
        diff_loss1 = torch.abs(gt_diff1 - pred_diff1)

        gt_diff2 = gt_layer[:, :, 1:] - gt_layer_2[:, :, :-1]
        pred_diff2 = pred_layer[:, :, 1:] - pred_layer_2[:, :, :-1]
        diff_loss2 = torch.abs(gt_diff2 - pred_diff2)

        # give more weight to the pixels with low difference
        diff_loss_weight1 = torch.sigmoid(-40 * (torch.abs(gt_diff1) - 0.1))
        diff_loss1 = diff_loss1 * diff_loss_weight1

        diff_loss_weight2 = torch.sigmoid(-40 * (torch.abs(gt_diff2) - 0.1))
        diff_loss2 = diff_loss2 * diff_loss_weight2

        # diff_loss1: [B, H-1, W]
        # diff_loss2: [B, H, W-1]

        # make them both the same size, fill with zeros
        diff_loss1 = torch.cat((diff_loss1, torch.zeros((diff_loss1.shape[0], 1, diff_loss1.shape[2]), device=diff_loss1.device)), dim=1)
        diff_loss2 = torch.cat((diff_loss2, torch.zeros((diff_loss2.shape[0], diff_loss2.shape[1], 1), device=diff_loss2.device)), dim=2)
        diff_loss =  diff_loss1 + diff_loss2

        return diff_loss
    
    def calculate_total_diff_loss(self, inputs, reconstructions):
        
        # total_diff_loss = torch.zeros_like(rec_loss)
        # for c1 in range(inputs.shape[1]):
        #     for c2 in range(inputs.shape[1]):
        #         diff_loss = self.calculate_diff_loss(reconstructions[:, c1, :, :], reconstructions[:, c2, :, :], inputs[:, c1, :, :], inputs[:, c2, :, :]) # [B, H, W]
        #         # total_diff_loss: [B, C, H, W]
        #         # diff_loss: [B, H, W]
        #         total_diff_loss[:, c1, :, :] = total_diff_loss[:, c1, :, :] + diff_loss
        # total_diff_loss = total_diff_loss / inputs.shape[1] 

        # do the same operation but this time store all tensors in the array and then combine them to get total_diff_loss
        
        total_diff_loss = []
        for c1 in range(inputs.shape[1]):
            diff_losses = []
            for c2 in range(inputs.shape[1]):
                diff_loss = self.calculate_diff_loss(reconstructions[:, c1, :, :], reconstructions[:, c2, :, :], inputs[:, c1, :, :], inputs[:, c2, :, :]) # [B, H, W]
                diff_losses.append(diff_loss.unsqueeze(1)) # [B, 1, H, W]

            diff_losses = torch.cat(diff_losses, dim=1) # [B, C, H, W]
            diff_losses = diff_losses.mean(dim=1) # [B, H, W]

            total_diff_loss.append(diff_losses.unsqueeze(1)) # [B, 1, H, W]
        total_diff_loss = torch.cat(total_diff_loss, dim=1) # [B, C, H, W]
        return total_diff_loss
    
    def relax_zero_pixels(self, rec_loss, inputs, reconstructions, zero_coefficient):
        # rec_loss: [B, C, H, W]
        # inputs: [B, C, H, W]
        # reconstructions: [B, C, H, W]
        # zero_coefficient: float
        # return: [B, C, H, W]

        zero_pixels = (inputs < 0.0001).float()

        boundary_values = reconstructions[:,0,:,:] # [B, H, W]

        lower_pixels_arr = []

        for c in range(reconstructions.shape[1]):
            current_layer = reconstructions[:,c,:,:]

            # compute pixels where current_layer is lower than boundary_values
            lower_pixels = (current_layer < boundary_values).float()

            lower_pixels_arr.append(lower_pixels.unsqueeze(1))

        lower_pixels = torch.cat(lower_pixels_arr, dim=1) # [B, C, H, W]

        zero_pixels = zero_pixels * lower_pixels

        rec_loss = rec_loss * (1 - zero_pixels) + zero_coefficient * rec_loss * zero_pixels

        # todo fix this LR gets lower when lots of the values are relaxed. need to apply more loss on non zero parts to preserve overall average learning rate.


        return rec_loss

    def calculate_l1_loss_ignoring_zero_pixels(self, inputs, reconstructions):
        # inputs: [B, C, H, W]
        # reconstructions: [B, C, H, W]
        # return: [B, C, H, W]

        inputs_flatten = inputs.view(-1) # [B * C * H * W]
        reconstructions_flatten = reconstructions.view(-1) # [B * C * H * W]

        non_zero_pixels = (inputs_flatten > 0.0001) # [B * C * H * W]

        non_zero_inputs = inputs_flatten[non_zero_pixels]
        non_zero_reconstructions = reconstructions_flatten[non_zero_pixels]

        l1_loss_ignoring_zero_pixels = torch.abs(non_zero_inputs - non_zero_reconstructions)

        return l1_loss_ignoring_zero_pixels

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        l1_loss_ignoring_zero_pixels = self.calculate_l1_loss_ignoring_zero_pixels(inputs, reconstructions)

        rec_loss = l1_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        
        if self.perceptual_weight >= -0.000001:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.zeros_like(rec_loss)
        
        total_diff_loss = self.calculate_total_diff_loss(inputs, reconstructions)
                
        rec_loss = (rec_loss + self.diff_loss_weight * total_diff_loss) / (1 + self.diff_loss_weight) # [B, C, H, W]

        rec_loss = self.relax_zero_pixels(rec_loss, inputs, reconstructions, 1 - self.relaxation_factor)

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/l1_loss".format(split): l1_loss.detach().mean(),
                   "{}/l1_loss_ignoring_zero_pixels".format(split): l1_loss_ignoring_zero_pixels.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                    "{}/p_loss".format(split): p_loss.detach().mean(),
                    "{}/total_diff_loss".format(split): total_diff_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


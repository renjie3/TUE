import numpy as np
import torch
from torch.autograd import Variable
from unsupervised.simclr.simclr_utils import train_simclr_noise_return_loss_tensor, get_linear_noise_csd_loss
from unsupervised.simsiam.simsiam_utils import train_simsiam_noise_return_loss_tensor
from unsupervised.moco.moco_utils import train_moco_noise_return_loss_tensor

import time

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        # print("device", device)
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    def min_min_attack_simsiam_return_loss_tensor_eot_v1(self, pos_samples_1, pos_samples_2, labels, model, random_noise=None, eot_size=30, ssl_weight=1, linear_noise_csd_weight=0, linear_noise_csd_index=1, k_grad=False):
    # v1 means it can repeat min_min_attack many times serially and average the results.
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):

            start = time.time()

            eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
            eot_loss = 0

            # perturb_org = torch.clamp(pos_samples_1.data + perturb, 0, 1)

            for i_eot in range(eot_size):
                time0 = time.time()
                
                perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                opt = torch.optim.SGD([perturb], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()

                if ssl_weight != 0:
                    simclr_loss = train_simsiam_noise_return_loss_tensor(model, perturb_img1, perturb_img2, k_grad=k_grad)
                else:
                    simclr_loss = 0

                if linear_noise_csd_weight != 0:
                    linear_noise_csd_loss = get_linear_noise_csd_loss(perturb, labels[:, linear_noise_csd_index])
                else:
                    linear_noise_csd_loss = 0

                # print(linear_xnoise_csd_loss)
                loss = simclr_loss * ssl_weight + linear_noise_csd_loss * linear_noise_csd_weight
                
                perturb.retain_grad()
                loss.backward()
                
                eot_grad += perturb.grad.data
                eot_loss += loss.item()
            
            eot_loss /= eot_size
            eot_grad /= eot_size

            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * eot_grad.sign() * (-1)
            sign_print = perturb.grad.data.sign() * (-1)
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            eta = (eta1 + eta2) / 2
            
            perturb = Variable(eta, requires_grad=True)
            
            # print("min_min_attack_simsiam_return_loss_tensor_eot_v1:", eot_loss)
            # input()

            end = time.time()

            # print("time: ", end - start)
        

        return eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_moco_return_loss_tensor_eot_v1(self, pos_samples_1, pos_samples_2, labels, model, random_noise=None, eot_size=30, ssl_weight=1, linear_noise_csd_weight=0, linear_noise_csd_index=1, k_grad=False):
    # v1 means it can repeat min_min_attack many times serially and average the results.
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):

            start = time.time()

            eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
            eot_loss = 0

            # perturb_org = torch.clamp(pos_samples_1.data + perturb, 0, 1)

            for i_eot in range(eot_size):
                time0 = time.time()
                
                perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                opt = torch.optim.SGD([perturb], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()

                if ssl_weight != 0:
                    simclr_loss = train_moco_noise_return_loss_tensor(model, perturb_img1, perturb_img2, k_grad=k_grad)
                else:
                    simclr_loss = 0

                if linear_noise_csd_weight != 0:
                    linear_noise_csd_loss = get_linear_noise_csd_loss(perturb, labels[:, linear_noise_csd_index])
                else:
                    linear_noise_csd_loss = 0

                loss = simclr_loss * ssl_weight + linear_noise_csd_loss * linear_noise_csd_weight
                
                perturb.retain_grad()
                loss.backward()
                
                eot_grad += perturb.grad.data
                eot_loss += loss.item()
            
            eot_loss /= eot_size
            eot_grad /= eot_size

            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * eot_grad.sign() * (-1)
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            # diff_eta = eta1 - eta2
            # print(diff_eta.cpu().numpy())
            eta = (eta1 + eta2) / 2
            
            perturb = Variable(eta, requires_grad=True)
            
            # print("min_min_attack_moco_return_loss_tensor_eot_v1:", eot_loss)
            # input()

            end = time.time()

            # print("time: ", end - start)
        

        return eta, train_loss_batch_sum / float(train_loss_batch_count)

    def min_min_attack_simclr_return_loss_tensor_eot_v1(self, pos_samples_1, pos_samples_2, labels, model, random_noise=None, temperature=None, eot_size=30, ssl_weight=1, linear_noise_csd_weight=0, linear_noise_csd_index=1):
        if random_noise is None:
            random_noise = torch.FloatTensor(*pos_samples_1.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb = Variable(random_noise, requires_grad=True)

        eta = random_noise
        train_loss_batch_sum, train_loss_batch_count = 0, 0
        for _ in range(self.num_steps):

            start = time.time()

            eot_grad = torch.zeros(perturb.shape, dtype=torch.float).to(device)
            eot_loss = 0

            for i_eot in range(eot_size):
                perturb_img1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
                perturb_img2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
                opt = torch.optim.SGD([perturb], lr=1e-3)
                opt.zero_grad()
                model.zero_grad()

                if ssl_weight != 0:
                    simclr_loss = train_simclr_noise_return_loss_tensor(model, perturb_img1, perturb_img2, temperature)
                else:
                    simclr_loss = 0

                if linear_noise_csd_weight != 0:
                    linear_noise_csd_loss = get_linear_noise_csd_loss(perturb, labels[:, linear_noise_csd_index])
                else:
                    linear_noise_csd_loss = 0

                loss = simclr_loss * ssl_weight + linear_noise_csd_loss * linear_noise_csd_weight
                
                perturb.retain_grad()
                loss.backward()
                
                eot_grad += perturb.grad.data
                eot_loss += loss.item()
            
            eot_loss /= eot_size
            eot_grad /= eot_size

            train_loss_batch = loss.item()/float(perturb.shape[0])
            train_loss_batch_sum += train_loss_batch * perturb.shape[0]
            train_loss_batch_count += perturb.shape[0]

            eta_step = self.step_size * eot_grad.sign() * (-1)
            perturb_img1 = perturb_img1.data + eta_step
            eta1 = torch.clamp(perturb_img1.data - pos_samples_1.data, -self.epsilon, self.epsilon)
            perturb_img2 = perturb_img2.data + eta_step
            eta2 = torch.clamp(perturb_img2.data - pos_samples_2.data, -self.epsilon, self.epsilon)
            eta = (eta1 + eta2) / 2
            perturb = Variable(eta, requires_grad=True)
            # print("min_min_attack_simclr_return_loss_tensor_eot_v1:", eot_loss)

            end = time.time()

            # print("time: ", end - start)

        return eta, train_loss_batch_sum / float(train_loss_batch_count)

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))

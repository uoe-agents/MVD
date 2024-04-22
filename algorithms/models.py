import torch
import torch.nn as nn
import utils

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, cfg):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = cfg.num_conv_layers
        self.num_filters = cfg.num_filters
        self.output_logits = False
        self.feature_dim = cfg.feature_dim

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2)])
        for i in range(1, self.num_layers):
            self.convs.extend([nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)])

        # get output shape
        x = torch.randn(*obs_shape).unsqueeze(0)
        conv = torch.relu(self.convs[0](x))
        for i in range(1, self.num_layers):
            conv = self.convs[i](conv)
        conv = conv.view(conv.size(0), -1)
        self.output_shape = conv.shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.output_shape, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.out_dim = self.feature_dim

        self.apply(utils.weight_init)

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        # h = conv.view(conv.size(0), -1)
        h = conv.reshape(conv.size(0), -1)
        return h

    def forward(self, obs, detach_encoder_conv=False, detach_encoder_head=False):
        h = self.forward_conv(obs)

        if detach_encoder_conv:
            h = h.detach()

        out = self.head(h)

        if not self.output_logits:
            out = torch.tanh(out)

        if detach_encoder_head:
            out = out.detach()
        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def copy_head_weights_from(self, source):
        """Tie head layers"""
        for i in range(2):
            utils.tie_weights(src=source.head[i], trg=self.head[i])

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)

class Decoder(nn.Module):
    def __init__(self, obs_shape, cfg):
        super().__init__()

        OUT_DIM = {2: 39, 4: 35, 6: 31}

        self.num_layers = cfg.num_conv_layers
        self.num_filters = cfg.num_filters
        if cfg.multi_view_disentanglement:
            self.feature_dim = cfg.feature_dim * 2
        else:
            self.feature_dim = cfg.feature_dim

        self.out_dim = OUT_DIM[self.num_layers]

        self.fc = nn.Linear(
            self.feature_dim, self.num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(self.num_filters, self.num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                self.num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

        self.apply(utils.weight_init)

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step):
        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)

class Actor(nn.Module):
    """torch.distributions implementation of a diagonal Gaussian policy."""
    def __init__(self, obs_shape, action_shape, cfg, proprioceptive_state_shape=None):
        super(Actor, self).__init__()

        self.cfg = cfg
        self.log_std_bounds = [cfg.actor_log_std_min, cfg.actor_log_std_max]
        self.algorithm = cfg.algorithm
        self.frame_stack = cfg.frame_stack
        self.multi_view_disentanglement = cfg.multi_view_disentanglement
        self.num_cameras = len(cfg.cameras)
        self.use_proprioceptive_state = cfg.use_proprioceptive_state
        self.encoder_output_dim = cfg.feature_dim
        self.device = cfg.device

        obs_shape = (int(obs_shape[0] / self.num_cameras), *obs_shape[1:])

        if self.multi_view_disentanglement:
            self.private_encoder = Encoder(obs_shape, cfg)
            self.shared_encoder = Encoder(obs_shape, cfg)
        else:
            self.encoder = Encoder(obs_shape, cfg)

        if self.multi_view_disentanglement:
            shared_feature_dim = self.shared_encoder.feature_dim
            private_feature_dim = self.private_encoder.feature_dim
            input_dim = shared_feature_dim + private_feature_dim
        else:
            input_dim = self.encoder.feature_dim

        if self.use_proprioceptive_state:
            input_dim += proprioceptive_state_shape[0]

        self.trunk = utils.mlp(input_dim, cfg.hidden_dim, 2 * action_shape[0], cfg.hidden_depth)

        self.outputs = dict()

        self.trunk.apply(utils.weight_init)

    def forward(self, obs, proprioceptive_state=None, detach_encoder_conv=False, detach_encoder_head=False, eval_on_single_cam=False):
        N = obs.shape[0]
        if eval_on_single_cam:
            num_cams = 1
        else:
            num_cams = self.num_cameras
        obs = obs.view((N * num_cams, -1, *obs.shape[2:]))
        if self.multi_view_disentanglement:
            z_private = self.private_encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)
            z_shared = self.shared_encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)
            z = torch.cat((z_shared, z_private), dim=-1)
        else:
            z = self.encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)

        if self.multi_view_disentanglement:
            z = z.view(N, num_cams, -1)
            z_shared = z_shared.view(N, num_cams, -1)
            z_private = z_private.view(N, num_cams, -1)

            idx_shared = torch.randperm(num_cams)[0]
            idx_private = torch.randperm(num_cams)[0]
            z_shared = z_shared[:, idx_shared]
            z_private = z_private[:, idx_private]
            z = torch.cat((z_shared, z_private), dim=-1)

        if self.use_proprioceptive_state:
            z = torch.cat((z, proprioceptive_state), dim=-1)

        mu, log_std = self.trunk(z).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist, z

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)

class Critic(nn.Module):
    """Critic network, employs double Q-learning."""
    def __init__(self, obs_shape, action_shape, cfg, proprioceptive_state_shape=None):
        super().__init__()

        self.frame_stack = cfg.frame_stack
        self.multi_view_disentanglement = cfg.multi_view_disentanglement
        self.num_cameras = len(cfg.cameras)
        self.use_proprioceptive_state = cfg.use_proprioceptive_state
        self.device = cfg.device

        obs_shape = (int(obs_shape[0] / self.num_cameras), *obs_shape[1:])

        if cfg.multi_view_disentanglement:
            self.private_encoder = Encoder(obs_shape, cfg)
            self.shared_encoder = Encoder(obs_shape, cfg)
        else:
            self.encoder = Encoder(obs_shape, cfg)

        if self.multi_view_disentanglement:
            shared_feature_dim = self.shared_encoder.feature_dim
            private_feature_dim = self.private_encoder.feature_dim
            input_dim = shared_feature_dim + private_feature_dim
        else:
            input_dim = self.encoder.feature_dim
        if self.use_proprioceptive_state:
            input_dim += proprioceptive_state_shape[0]

        self.Q1 = utils.mlp(input_dim + action_shape[0],
                            cfg.hidden_dim, 1, cfg.hidden_depth)
        self.Q2 = utils.mlp(input_dim + action_shape[0],
                            cfg.hidden_dim, 1, cfg.hidden_depth)

        self.outputs = dict()

        self.Q1.apply(utils.weight_init)
        self.Q2.apply(utils.weight_init)

    def forward(self, obs, action, proprioceptive_state=None, detach_encoder_conv=False, detach_encoder_head=False):
        N = obs.shape[0]
        obs = obs.view((N * self.num_cameras, -1, *obs.shape[2:]))
        if self.multi_view_disentanglement:
            z_shared = self.shared_encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)
            z_private = self.private_encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)
        else:
            z = self.encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)

        if self.multi_view_disentanglement:
            z_shared = z_shared.view(N, self.num_cameras, -1)
            z_private = z_private.view(N, self.num_cameras, -1)
            idx_shared = torch.randperm(self.num_cameras)[0]
            idx_private = torch.randperm(self.num_cameras)[0]
            z_shared = z_shared[:, idx_shared]
            z_private = z_private[:, idx_private]
            z = torch.cat((z_shared, z_private), dim=-1)

        assert z.size(0) == action.size(0)

        if self.use_proprioceptive_state:
            z = torch.cat((z, proprioceptive_state), dim=-1)

        obs_action = torch.cat([z, action], dim=-1)

        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def get_representation(self, obs, detach_encoder_conv=False, detach_encoder_head=False):
        N = obs.shape[0]
        obs = obs.view((N * self.num_cameras, -1, *obs.shape[2:]))
        if self.multi_view_disentanglement:
            z_shared = self.shared_encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)
            z_private = self.private_encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)
            z = torch.cat((z_shared, z_private), dim=-1)
        else:
            z = self.encoder(obs, detach_encoder_conv=detach_encoder_conv, detach_encoder_head=detach_encoder_head)
            z_shared = None
            z_private = None

        if self.multi_view_disentanglement:
            z_private = z_private.reshape(N, self.num_cameras, -1)
            z_shared = z_shared.reshape(N, self.num_cameras, -1)

        z = z.reshape(N, self.num_cameras, -1)

        return z, z_shared, z_private

    def log(self, logger, step):
        if self.multi_view_disentanglement:
            self.shared_encoder.log(logger, step)
            self.private_encoder.log(logger, step)
        else:
            self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
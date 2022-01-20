
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from utils.networks import Model, MLP, Mode

class BCAgent(Model):
    def __init__(self, args):
        super().__init__()
        n_inputs = 1024*10 + 32*8
        self.n_outputs = 200
        self.model = MLP(
            n_inputs=n_inputs,
            n_outputs=self.n_outputs,
            n_layers=4,
            layer_size=1024,
            dropout=0.1
        ).to(self.device)

    def forward(self, seg_tokens, class_tokens, ner_probs):
        state_tensor = torch.cat((
            seg_tokens.flatten(start_dim=1),
            class_tokens.flatten(start_dim=1),
            ner_probs.flatten(start_dim=1),
        ), dim=-1)
        output = self.model(state_tensor)
        return output

    def act(self, state, deterministic=True):
        seg_tokens = torch.LongTensor(state['seg_tokens']).to(self.device).unsqueeze(0)
        class_tokens = torch.LongTensor(state['class_tokens']).to(self.device).unsqueeze(0)
        ner_probs = torch.FloatTensor(state['ner_probs']).to(self.device).unsqueeze(0)
        with Mode(self, 'eval'):
            with torch.no_grad():
                output = self(seg_tokens, class_tokens, ner_probs)
                probs = F.softmax(output, dim=-1).squeeze(0).cpu().numpy()
        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(range(self.n_outputs), p=probs)
        return action

    def learn(self, dataset, env):
        dataloader = DataLoader(dataset,
                                num_workers=4,
                                batch_size=64,
                                sampler=RandomSampler(dataset))
        epochs = 1000
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        for epoch in range(1, epochs + 1):
            for step, (seg_tokens, class_tokens, ner_probs, action) in enumerate(dataloader):
                seg_tokens = seg_tokens.to(self.device)
                class_tokens = class_tokens.to(self.device)
                ner_probs = ner_probs.to(self.device)
                action = action.to(self.device).long()
                output = self(seg_tokens, class_tokens, ner_probs)
                loss = F.cross_entropy(output, action)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print(loss.item())
            self.evaluate(env)


    def evaluate(self, env, n_episodes=1):
        rewards = []
        for ep in range(1, n_episodes + 1):
            state = env.reset()
            preds = env.essay.correct_predictions
            done = False
            pred_idx = 0
            while not done:
                action = self.act(state)
                if pred_idx < len(preds):
                    print(action, len(preds[pred_idx]))
                pred_idx += 1
                state, reward, done, info = env.step(action)
            rewards.append(env.current_state_value())
        metrics = {
            'Average Eval Reward': sum(rewards) / len(rewards)
        }
        print(metrics)
        return metrics
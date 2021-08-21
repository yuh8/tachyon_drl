import numpy as np
import tensorflow as tf
from data_gen_rl import generate_smile
from src.reward_utils import get_terminal_reward, get_padded_reward_vec, is_valid_smile
from src.misc_utils import load_json_model
from src.CONSTS import MOL_DICT, ALPHA, BATCH_SIZE_RL, EMBEDDING_SIZE_GEN


def loss_function(distributed_reward_with_idx, action_logits):
    action_probs = tf.nn.softmax(action_logits, axis=-1)
    # [BATCH, MAX_MOL_LEN]
    distributed_reward = distributed_reward_with_idx[:, 0, :]
    # [BATCH, MAX_MOL_LEN, 1]
    distributed_reward = tf.expand_dims(distributed_reward, axis=-1)
    # [BATCH, MAX_MOL_LEN]
    reward_idx = tf.cast(distributed_reward_with_idx[:, 1, :], tf.int32)
    # [BATCH, MAX_MOL_LEN, MOL_DICT_LEN + 1]
    reward_onehot = tf.one_hot(reward_idx, len(MOL_DICT) + 1,
                               on_value=1, off_value=0, axis=-1)
    reward_onehot = tf.cast(reward_onehot, distributed_reward.dtype)
    distributed_reward = distributed_reward * reward_onehot
    # [BATCH, MAX_MOL_LEN]
    mask = tf.math.less(reward_idx, len(MOL_DICT))
    # [BATCH, MAX_MOL_LEN, 1]
    mask = tf.expand_dims(mask, axis=-1)
    # [BATCH, MAX_MOL_LEN, MOL_DICT_LEN + 1]
    reward = action_probs * distributed_reward
    negative_entropy = action_probs * ALPHA * tf.math.log(action_probs)
    loss_ = negative_entropy - reward
    mask = tf.cast(mask, loss_.dtype)
    loss_ *= mask
    # [BATCH, MAX_MOL_LEN]
    loss_ = tf.reduce_sum(loss_, axis=-1)
    # [BATCH]
    loss_ = tf.reduce_sum(loss_, axis=-1)
    # Scalar
    loss_ = tf.reduce_mean(loss_)
    return loss_


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model=EMBEDDING_SIZE_GEN, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
        }
        return config


def get_optimizer():
    opt_op = tf.keras.optimizers.Adam(learning_rate=CustomSchedule())
    return opt_op


def create_policy_model(model_path):
    gen_rl = load_json_model(model_path)
    gen_rl.compile(optimizer=get_optimizer(), loss=loss_function)
    return gen_rl


def create_predict_model(model_path):
    pred_net = load_json_model(model_path)
    pred_net.compile(optimizer=get_optimizer(), loss='mse')
    return pred_net


if __name__ == "__main__":
    gen_rl = create_policy_model("generator_model/generator_model.json")
    pred_net = create_predict_model("predictor_model/predictor_model.json")
    breakpoint()
    gen_rl.load_weights('./generator_weights/generator')
    pred_net.load_weights('./predictor_weights/predictor')
    pred_net.trainable = False
    smi_bank = []
    running_reward = 0
    running_validity = 0
    for ii in range(10000):
        input_batch = []
        target_batch = []
        batch_reward = 0
        batch_validity = 0
        for _ in range(BATCH_SIZE_RL):
            start_tokens, start_tokens_ids, generated_tokens, generated_token_ids = generate_smile(gen_rl)
            if generated_tokens[-1] == "E":
                smi = "".join(generated_tokens[:-1])
            else:
                smi = "".join(generated_tokens)

            if is_valid_smile(smi):
                batch_validity += 1
                if len(smi_bank) >= 20:
                    smi_bank = smi_bank[:-1]
                    smi_bank.insert(0, smi)
                else:
                    smi_bank.append(smi)

            r, T = get_terminal_reward(generated_tokens, smi_bank, pred_net)
            r_vec = get_padded_reward_vec(r, 0.99, T)
            distributed_reward_with_idx = np.vstack([r_vec, np.array(generated_token_ids)])
            input_batch.append(generated_token_ids)
            target_batch.append(distributed_reward_with_idx)
            batch_reward += r
        running_reward = 0.1 * batch_reward + (1 - 0.1) * running_reward
        input_batch = np.vstack(input_batch)
        target_batch = np.stack(target_batch)
        loss = gen_rl.train_on_batch(input_batch, target_batch)
        print("train_loss={0}, reward={1} and validity={2} at iteration {3}".format(np.round(loss, 2),
                                                                                    np.round(running_reward, 2),
                                                                                    np.round(batch_validity / BATCH_SIZE_RL, 2), ii))
        if ii % 10 == 0:
            gen_rl.save_weights('./reinforce_weights/reinforce')

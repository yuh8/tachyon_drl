from src.data_process_utils import tokenize_smi, get_encoded_smi_rl
from train_generator_rl import create_predict_model
from src.CONSTS import Y_MIN, Y_MAX


smi = "NC(=O)c1ccc2c3c1OC1C(O)CCC4(O)C(C2)N(CC2CCC2)CCC314"
smi_token_list = tokenize_smi(smi)
smi_bank = []
pred_net = create_predict_model("predictor_model/predictor_model.json")
pred_net.load_weights('./predictor_weights/predictor')


encoded_smi = get_encoded_smi_rl(smi_token_list)
scaled_reward = pred_net(encoded_smi, training=False).numpy()[0][0]
reward = scaled_reward * (Y_MAX - Y_MIN) + Y_MIN
breakpoint()

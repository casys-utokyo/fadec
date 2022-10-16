import time
import numpy as np
import nngen as ng

from keyframe_buffer import KeyframeBuffer
from utils import lstm_state_calculator

class Verifier():

    def __init__(self, inputs, outputs, max_n_measurement_frames, fusion, act_dtype):

        self.inputs = inputs
        self.outputs = outputs
        self.max_n_measurement_frames = max_n_measurement_frames
        self.fusion = fusion
        self.act_dtype = act_dtype

        self.files = ["layer1", "layer2", "layer3", "layer4", "layer5",
                      "feature_one_sixteen", "feature_one_eight", "feature_quarter", "feature_half",
                      "cost_volume",
                      "skip0", "skip1", "skip2", "skip3", "bottom",
                      "cell_state", "hidden_state",
                      "depth_org"]
        self.shifts = [11, 11, 11, 12, 13,
                       11, 11, 10, 9,
                       7,
                       13, 13, 13, 12, 13,
                       12, 14,
                       14]


    def prepare_input_value(self, value, lshift):
        act_dtype = self.act_dtype
        ret = value * (1 << lshift)
        ret = np.clip(ret, -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
        return np.round(ret.astype(np.float64)).astype(np.int64)


    def calc_depth(self, depth_org):
        min_depth = 0.25
        max_depth = 20.0
        inverse_depth_base = 1 / max_depth
        inverse_depth_multiplier = 1 / min_depth - 1 / max_depth

        inverse_depth_full = inverse_depth_multiplier * depth_org + inverse_depth_base
        return 1.0 / inverse_depth_full


    def print_results(self, eval_outs, idx, verbose):
        outputs = self.outputs
        files = self.files
        shifts = self.shifts
        for i in range(len(eval_outs)):
            if not(verbose) and i != len(eval_outs) - 1:
                continue
            if len(eval_outs) == len(files) and i < 9:
                ground_truth = outputs[files[i]][idx+1]
            else:
                ground_truth = outputs[files[i]][idx]
            print(files[i], ground_truth.shape)
            output_layer_value = eval_outs[i].transpose(0, 3, 1, 2) / (1 << shifts[i])
            print(np.mean(output_layer_value.reshape(-1)), np.std(output_layer_value.reshape(-1)))
            print(np.mean(ground_truth.reshape(-1)), np.std(ground_truth.reshape(-1)))
            print(np.corrcoef(output_layer_value.reshape(-1), ground_truth.reshape(-1))[0, 1])
            print("--------------------------")
        print()


    def verify_all(self, layers, reference_features, cost_volume, skips, lstm_states, depth_full, verbose=False):
        inputs = self.inputs
        max_n_measurement_frames = self.max_n_measurement_frames
        fusion = self.fusion
        shifts = self.shifts
        prepare_input_value = self.prepare_input_value
        calc_depth = self.calc_depth
        print_results = self.print_results

        test_keyframe_buffer_size = 30
        test_keyframe_pose_distance = 0.1
        test_optimal_t_measure = 0.15
        test_optimal_R_measure = 0.0
        keyframe_buffer = KeyframeBuffer(buffer_size=test_keyframe_buffer_size,
                                         keyframe_pose_distance=test_keyframe_pose_distance,
                                         optimal_t_score=test_optimal_t_measure,
                                         optimal_R_score=test_optimal_R_measure,
                                         store_return_indices=False)

        reference_images = inputs["reference_image"]
        reference_poses = inputs["reference_pose"]
        n_measurement_frames = inputs["n_measurement_frames"]

        start_time = time.process_time()

        lstm_state = None
        previous_depth = None
        previous_pose = None
        calc = lstm_state_calculator(inputs, prepare_input_value, 14-1, 12)
        predictions = {"reference_image": [], "feature_half": [], "measurement_features": [], "hidden_state": [], "cell_state": [], "prediction": []}
        idx = 0
        for n in range(len(reference_images)):
            response = keyframe_buffer.try_new_keyframe(reference_poses[n][0])

            print("evaluating %05d.png (response: %d) ..." % (n + 3, response))

            if response == 2 or response == 4 or response == 5:
                continue
            elif response == 3:
                previous_depth = None
                previous_pose = None
                lstm_state = None
                continue

            ng_inputs = {}
            reference_image_value = prepare_input_value(reference_images[n].transpose(0, 2, 3, 1), 12)
            ng_inputs["reference_image"] = reference_image_value
            predictions["reference_image"].append(reference_image_value.copy())

            if response == 0:
                eval_outs = ng.eval(layers + reference_features[::-1], **ng_inputs)
                keyframe_buffer.add_new_keyframe(reference_poses[n][0], eval_outs[len(layers)+3])
                predictions["feature_half"].append(eval_outs[len(layers)+3].copy())
                print_results(eval_outs, idx, verbose)
                continue

            measurement_frames = keyframe_buffer.get_best_measurement_frames(reference_poses[n][0], max_n_measurement_frames)
            measurement_features_value = []
            for measurement_frame in measurement_frames:
                measurement_features_value.append(measurement_frame[1])
            for _ in range(max_n_measurement_frames - len(measurement_features_value)):
                measurement_features_value.append(np.zeros_like(measurement_features_value[0]))
            hidden_state_value, cell_state_value = calc(lstm_state, previous_depth, previous_pose, reference_poses[n])

            fusion.prep(idx, len(measurement_frames), measurement_features_value)
            ng_inputs["hidden_state"] = hidden_state_value
            ng_inputs["cell_state"] = cell_state_value


            input_layer_values = ng_inputs
            output_layers = layers + reference_features[::-1] + cost_volume + skips + lstm_states[::-1] + depth_full
            eval_outs = ng.eval(output_layers, **ng_inputs)
            output_layer_values = dict(zip(self.files, eval_outs))

            lstm_state = eval_outs[-2], eval_outs[-3]
            keyframe_buffer.add_new_keyframe(reference_poses[n][0], eval_outs[len(layers)+3])
            previous_depth = calc_depth((eval_outs[-1].transpose(0, 3, 1, 2) / (1 << shifts[-1])).astype(np.float32))
            previous_pose = reference_poses[n].copy()

            predictions["prediction"].append(previous_depth.copy())
            predictions["feature_half"].append(eval_outs[len(layers)+3].copy())
            predictions["measurement_features"].append(measurement_features_value.copy())
            predictions["hidden_state"].append(hidden_state_value.copy())
            predictions["cell_state"].append(cell_state_value.copy())

            print_results(eval_outs, idx, verbose)

            idx += 1

        print("\t%f [s]" % (time.process_time() - start_time))
        return input_layer_values, output_layer_values, output_layers, predictions


    def verify_one(self, layers, reference_features, cost_volume, skips, lstm_states, depth_full, verbose=False):
        inputs = self.inputs
        outputs = self.outputs
        fusion = self.fusion
        prepare_input_value = self.prepare_input_value
        calc_depth = self.calc_depth
        print_results = self.print_results

        reference_images = inputs["reference_image"]
        reference_poses = inputs["reference_pose"]
        n_measurement_frames = inputs["n_measurement_frames"]
        measurement_features = inputs["measurement_features"]

        start_time = time.process_time()

        calc = lstm_state_calculator(inputs, prepare_input_value, 14-1, 12)

        n = 9
        prev_n = n - 3
        idx = 1

        print("evaluating %05d.png ..." % (n + 3))

        ng_inputs = {}
        reference_image_value = prepare_input_value(reference_images[n].transpose(0, 2, 3, 1), 12)
        ng_inputs["reference_image"] = reference_image_value

        measurement_features_value = prepare_input_value(measurement_features[idx].transpose(0, 1, 3, 4, 2), 9)

        lstm_state = inputs["hidden_state"][idx], inputs["cell_state"][idx]
        previous_depth = calc_depth(outputs["depth_org"][idx-1])
        previous_pose = reference_poses[prev_n]
        hidden_state_value, cell_state_value = calc(lstm_state, previous_depth, previous_pose, reference_poses[n])

        fusion.prep(idx, n_measurement_frames[idx], measurement_features_value)
        ng_inputs["hidden_state"] = hidden_state_value
        ng_inputs["cell_state"] = cell_state_value


        input_layer_values = ng_inputs
        output_layers = layers + reference_features[::-1] + cost_volume + skips + lstm_states[::-1] + depth_full
        eval_outs = ng.eval(output_layers, **ng_inputs)
        output_layer_values = dict(zip(self.files, eval_outs))

        print_results(eval_outs, idx, verbose)

        print("\t%f [s]" % (time.process_time() - start_time))
        return input_layer_values, output_layer_values, output_layers

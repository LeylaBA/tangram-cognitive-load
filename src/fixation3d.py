import numpy as np
import pandas as pd


def is_fixation_idt(gaze_points, fixation_size):
    X, Y, Z = gaze_points["X"], gaze_points["Y"], gaze_points["Z"]
    if max([max(X) - min(X), max(Y) - min(Y), max(Z) - min(Z)]) < fixation_size:
        return True
    return False


def extract_fixations(
    data, min_points_per_fixation=2, min_fixation_size=0.047, max_fixation_size=0.095
):
    """Detect fixations in the gaze data using the I-DT algorithm. The defaults were determined empirically from the data provided by Aziz (2022), using a Hololens 2.
    Make sure to adjust the parameters depending on your task and setup for optimal results.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing gaze points in 3D.
    min_points_per_fixation : int, optional
        The minimum number of points required to start counting a fixation. Determined by the minimum duration of a fixation (~0.08 s)
        multiplied by the sampling rate of the eye tracker.
    min_fixation_size : int, optional
        The minimum size of a fixation in pixels.
    max_fixation_size : int, optional
        The maximum size of a fixation in pixels.
    """
    timestamps = data["timestamp"].values
    gaze_x = data["direction_x"].values
    gaze_y = data["direction_y"].values
    gaze_z = data["direction_z"].values
    gaze_origin = data[["origin_x", "origin_y", "origin_z"]].values

    gaze_points_queue = {"timestamps": [], "X": [], "Y": [], "Z": [], "origin": []}
    fixation = {"timestamps": [], "X": [], "Y": [], "Z": [], "origin": []}
    fixations = []
    processing_fixation = False

    for i in range(len(timestamps)):
        if not processing_fixation:
            # add points to the queue until we have enough points to start counting a fixation
            gaze_points_queue["timestamps"].append(timestamps[i])
            gaze_points_queue["X"].append(gaze_x[i])
            gaze_points_queue["Y"].append(gaze_y[i])
            gaze_points_queue["Z"].append(gaze_z[i])
            gaze_points_queue["origin"].append(gaze_origin[i])

            if len(gaze_points_queue["timestamps"]) >= min_points_per_fixation:
                # found enough points for a fixation, keep the newest min_points_per_fixation points
                for key in gaze_points_queue.keys():
                    gaze_points_queue[key] = gaze_points_queue[key][
                        -min_points_per_fixation:
                    ]

                if is_fixation_idt(gaze_points_queue, min_fixation_size):
                    processing_fixation = True
                    for key in gaze_points_queue.keys():
                        fixation[key] = gaze_points_queue[key]

        else:
            # add points to the fixation
            fixation["timestamps"].append(timestamps[i])
            fixation["X"].append(gaze_x[i])
            fixation["Y"].append(gaze_y[i])
            fixation["Z"].append(gaze_z[i])
            fixation["origin"].append(gaze_origin[i])

            if not is_fixation_idt(fixation, max_fixation_size):
                # fixation ended, put the outlier back to an empty queue
                gaze_points_queue = {
                    "timestamps": [],
                    "X": [],
                    "Y": [],
                    "Z": [],
                    "origin": [],
                }
                for key in fixation.keys():
                    gaze_points_queue[key].append(fixation[key].pop())

                # save fixation
                start_timestamp = fixation["timestamps"][0]
                end_timestamp = fixation["timestamps"][-1]
                x = np.mean(fixation["X"])
                y = np.mean(fixation["Y"])
                z = np.mean(fixation["Z"])
                origin = np.array(fixation["origin"])
                origin_x = np.mean(origin[:, 0])
                origin_y = np.mean(origin[:, 1])
                origin_z = np.mean(origin[:, 2])
                fixations.append(
                    [
                        start_timestamp,
                        end_timestamp,
                        x,
                        y,
                        z,
                        origin_x,
                        origin_y,
                        origin_z,
                    ]
                )

                # reset fixation
                fixation = {"timestamps": [], "X": [], "Y": [], "Z": [], "origin": []}
                processing_fixation = False

    fixation_df = pd.DataFrame(
        fixations,
        columns=["start", "end", "x", "y", "z", "origin_x", "origin_y", "origin_z"],
    )
    return fixation_df

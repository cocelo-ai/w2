import csv
import os
import datetime
import logging


class Logger:
    def __init__(self):
        # 1) log level mapping
        log_level_dict = {
            "DEBUG":    logging.DEBUG,
            "INFO":     logging.INFO,
            "WARNING":  logging.WARNING,
            "ERROR":    logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "FATAL":    logging.FATAL,
        }
        LOG_LEVEL = "DEBUG"
        MAX_NUM_LOG = 20

        VERBOSE = False
        SAVE_DATA = False
        SAVE_DATA_FILENAME = "dataset_250525_4.csv"
        SAVE_FREQ = 100
        DATA_FIELDS = [
            "left_hip_current_position", "left_hip_target_position", "left_hip_current_velocity", "left_hip_torque",
            "right_hip_current_position", "right_hip_target_position", "right_hip_current_velocity", "right_hip_torque",
            "left_shoulder_current_position", "left_shoulder_target_position", "left_shoulder_current_velocity",
            "left_shoulder_torque",
            "right_shoulder_current_position", "right_shoulder_target_position", "right_shoulder_current_velocity",
            "right_shoulder_torque",
            "left_leg_current_position", "left_leg_target_position", "left_leg_current_velocity", "left_leg_torque",
            "right_leg_current_position", "right_leg_target_position", "right_leg_current_velocity", "right_leg_torque",
            "left_wheel_current_velocity", "left_wheel_target_velocity", "left_wheel_current_sin_position",
            "left_wheel_current_cos_position", "left_wheel_torque",
            "right_wheel_current_velocity", "right_wheel_target_velocity", "right_wheel_current_sin_position",
            "right_wheel_current_cos_position", "right_wheel_torque"
        ]

        # (1) debug properties
        self.save_freq     = int(SAVE_FREQ)
        self.verbose       = VERBOSE
        self.level         = log_level_dict[LOG_LEVEL]
        self.save_data_file = None

        # (2) logger setup
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_root = os.path.join(base_dir, '..', 'log')
        os.makedirs(log_root, exist_ok=True)

        # --- Limit number of log files to 1000 (by creation time) ---
        existing = [
            os.path.join(log_root, f)
            for f in os.listdir(log_root)
            if f.endswith('.log')
        ]
        existing.sort(key=lambda p: os.path.getctime(p))
        while len(existing) >= MAX_NUM_LOG:
            os.remove(existing.pop(0))
        # --------------------------------------------

        log_start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join(log_root, log_start_time + '.log')

        try:
            # create new log file and write header
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("Log file created at: " + log_start_time + "\n")

            self.logger = logging.getLogger("debugger")
            logging.basicConfig(
                filename=log_path,
                format='(%(asctime)s) %(levelname)s: %(message)s',
                datefmt ='%y/%m/%d %H:%M:%S',
                level=self.level
            )

        except Exception as e:
            print(f"Cannot open log file at {log_path}")
            raise e

        # (3) buffer
        self.buffer = [None for _ in range(self.save_freq)]
        self.idx    = 0

        # (4) open CSV file for SAVE_DATA
        if SAVE_DATA:
            try:
                csv_dir = os.path.join(base_dir, '..', 'dataset', SAVE_DATA_FILENAME)
                self.save_data_file, self.data_writer = self.open(path=csv_dir, columns=DATA_FIELDS)
            except Exception as e:
                print("Cannot open save_data_file at:", SAVE_DATA_FILENAME)
                raise e

        self.info("Logger initialized")

    def open(self, path, columns):
        try:
            exist_flag = os.path.exists(path)
            _file = open(path, 'a', newline='')  # Keep file handler open
            writer = csv.writer(_file)
            if not exist_flag:
                writer.writerow(columns)
                _file.flush()
            return _file, writer
        except Exception as e:
            self.error(f"The logger failed to open the files due to the following error: {e}")
            raise e

    def close(self):
        if self.save_data_file is not None:
            self.save_data_file.close()

    def debug(self, log):
        self.logger.debug(log)
        if self.verbose is True and self.level <= logging.DEBUG:
            print("[DEBUG] " + log)

    def info(self, log):
        self.logger.info(log)
        if self.verbose is True and self.level <= logging.INFO:
            print("[INFO] " + log)

    def warning(self, log):
        self.logger.warning(log)
        if self.verbose is True and self.level <= logging.WARNING:
            print("[WARNING] " + log)

    def error(self, log):
        self.logger.error(log)
        if self.verbose is True and self.level <= logging.ERROR:
            print("[ERROR] " + log)

    def critical(self, log):
        self.logger.critical(log)
        if self.verbose is True and self.level <= logging.CRITICAL:
            print("[CRITICAL] " + log)

    def save(self):
        try:
            self.data_writer.writerows(self.buffer)
            self.save_data_file.flush()  # Ensure data is written to disk
            self.info("The csv is saved")
        except Exception as e:
            self.error(f"The logger failed to save the data due to the following error: {e}")

    def data_log(self, data_row):
        if self.save_data_file is not None:
            self.buffer[self.idx] = data_row.tolist()
            self.idx += 1
            if self.idx >= self.save_freq:
                self.save()
                self.idx = 0

# Created by zpehlivan at 13/02/2020
import json
import logging
import logging.config
import subprocess
import sys



if __name__ == '__main__':

    config_path = sys.argv[1]
    options = []
    try:
        with open(config_path) as json_data_file:
            options = json.load(json_data_file)
    except:
        print("Reading config file problem")

    if len(options) == 0:
        print("No config file found. Exit !!!")
        exit(1)
    else :
        logging_conf_path = options["logging"]
        logging.config.fileConfig(logging_conf_path)
        log = logging.getLogger("root")
        log.info("Launching docker temp application...")
        script = "python {} {}".format(options["script"], options["params"])
        subprocess.run(script, shell=True)

        log.info("OVER OVER")
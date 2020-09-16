import time
import subprocess


def run(cmd):
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            )
    stdout, stderr = proc.communicate()

    return proc.returncode, stdout, stderr


#code, out, err = run(['/usr/bin/nvidia-smi'])
#print(code)
#print(out)
#print(err)


with open("nvidia-smi-out.txt", "w+") as outf:

    while True:

        _, out, _ = run(['/usr/bin/nvidia-smi'])
        outf.write(out)
        outf.write("\n")

        _, out, _ = run(['/usr/bin/sensors'])
        outf.write(out)

        outf.flush()
        time.sleep(5*60)

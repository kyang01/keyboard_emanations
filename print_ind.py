import time

fname = 'alice.txt'

last_len = -1

for i in range(5):
	print 5 - i
	time.sleep(1)

start_time = time.time()
with open(fname, "w") as f_out:
	while True:
		f = open(fname, 'r') 
		txt = f.read()
		new_len = len(txt) - 1
		if new_len != last_len:
			last_char = ""
			if len(txt) > 0:
				last_char = txt[-1]
			else:
				last_char = "none"
			print last_char, new_len, time.time() - start_time
			f_out.write(last_char + ", " + str(new_len) + "," + str(time.time() - start_time) + "\n")
			last_len = new_len
		time.sleep(0.01)
import time

# file to read from
in_fname = 'input_text.txt'

# file to save from
out_fname = 'output_text.txt'


def get_len(fname):
	'''
		Gets the length and text for file fname
	'''
	# re open the infile
	f = open(fname, 'r') 
	txt = f.read()

	# new length and change
	new_len = len(txt)

	return new_len, txt

# the last known file length
last_len, last_txt = get_len(in_fname)


# countdown before start
for i in range(5):
	print 5 - i
	time.sleep(1)

# get the time at the start
start_time = time.time()

# file to write out
with open(out_fname, "w") as f_out:
	while True:
		time.sleep(0.01)
		new_len, new_txt = get_len(in_fname)
		
		# see if the file has changed
		len_change = new_len - last_len

		if len_change == 0:
			continue

		if len_change > 0:
			txt_chng = new_txt[-len_change:]
		elif len_change < 0:
			txt_chng = '###DEL###' + last_txt[len_change:]


		tm = time.time() - start_time
		result = txt_chng + ", " + str(tm) + "\n"
		print(result) 
		f_out.write(result)
		
		# save for next time
		last_len, last_txt = new_len, new_txt
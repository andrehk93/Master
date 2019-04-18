import csv
import os

print("Reading .csv file...")

folder = 'raw'

if not os.path.exists(folder):
	os.makedirs(folder)

with open('india-news-headlines.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	metaline = reader.__next__()

	status = 0
	update = 50000

	print("Starting writing to file...")
	done = False
	while not done:
		try:
			row = reader.__next__()
			if len(row) == 0:
				done = True
			id_tag, category, headline = row

			category = category.replace(".", "_")

			if (not os.path.exists(os.path.join(folder, category))):
				os.makedirs(os.path.join(folder, category))

			with open(os.path.join(folder, category, id_tag), "w") as file:
				file.write(headline)

			status += 1

			if (status % update == 0):
				print("Currently written " + str(status) + " files")
		except UnicodeDecodeError as e:
			pass

print("Successfully written all data to file!")
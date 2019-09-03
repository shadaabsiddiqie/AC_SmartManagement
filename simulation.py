import Queue,time
import random
import numpy as np
import math 
import csv
class Cell:
	def __init__(self,row,col,grid,initial_temp=23,final_temp=14,hasac=False,heat_capacity=0.7):
		self.row = row
		self.col = col
		self.temp = initial_temp
		self.final_temp = final_temp
		self.hasac = hasac
		self.cutoff = 1
		self.power = 0
		self.eff = 0
		self.clock_value = 0.1
		self.heat_capacity = 700
		self.grid = grid

	def set_temp(self,temp) :
		self.temp = temp

	def set_final_temp(self,temp):
		self.final_temp = temp

	def add_ac(self,cutoff_temp=15,power=5507,eff=3.5):
		self.hasac = True
		self.cutoff_temp = cutoff_temp
		self.power = power
		self.eff = eff

	def update_cell(self):
		self.eff = random.randint(25,35)/10
		region = self.grid.regions[self.row][self.col]
		temp = 0
		count = 0
		for pos in self.grid.region_list[region]:
			temp += self.grid.grid[pos[0]][pos[1]].temp
			count += 1
		temp = float(temp)/float(count)
		
		# if temp - self.cutoff_temp <= self.cutoff and self.temp <= 0 :
		# 	if self.hasac :
		# 		#print temp , self.cutoff_temp
		# 		# print "AC SWITCHED OFF " +str(self.row) +" "+str(self.col)
		# 		self.hasac = False 

		if temp - self.cutoff_temp >= self.cutoff :
			if not self.hasac:
				#print temp , self.cutoff_temp
				# print "AC SWITCHED ON "+str(self.row) +" "+str(self.col)
				self.hasac = True

		if self.hasac :
			energy = self.power * self.eff * self.clock_value
			temp_diff = energy / self.heat_capacity
			if(self.temp >0):
				self.temp -= temp_diff
class GRID:
	def __init__(self,rows,cols,default_temp=23):
		self.grid = [[Cell(i,j,self,default_temp) for j in range(cols)] for i in range(rows)]
		self.ac_cells = []
		self.k = 0.873
		self.rows = rows
		self.cols = cols
		self.regions = [[1e9 for j in range(cols)]for i in range(rows)]
		self.region_count = 0
		self.region_cutoff = 3
		self.region_max = {}
		self.region_min = {}
		self.vis = {}
		self.region_size = {}
		self.region_list = {}
		self.regionminsize = 18
		self.regionmaxsize = 20
		self.total_power = 0

	def add_ac(self,row,col,cutoff_temp=15,power=5507,eff=3.5):
		self.grid[row][col].add_ac(cutoff_temp,power,eff)
		self.ac_cells.append(self.grid[row][col])

	def set_final_temp(self,row,col,temp):
		self.grid[row][col].set_final_temp(temp)


	def start_distribution(self):
		for i in range(self.rows):
			for j in range(self.cols):
				if self.regions[i][j] == 1e9 :
					self.get_distribution(i,j,self.region_count+1)
					self.region_count += 1

		for i in range(self.rows):
			for j in range(self.cols):
				if self.region_size[self.regions[i][j]] == 1 :
					self.region_size[self.regions[i][j]] -= 1
					# if j+1 < self.cols:
					self.region_size[self.regions[i][j+1]] += 1
					self.regions[i][j] = self.regions[i][j+1]
					self.region_count -= 1

		for i in range(1,self.region_count+1):
			self.region_list[i] = []
		
		for i in range(self.rows):
			for j in range(self.cols):
				# if (self.regions[i][j] in range(1,self.region_count+1) ):
				self.region_list[self.regions[i][j]].append((i,j))
		
		self.place_acs()

	def place_acs(self):
		for key in self.region_list.keys():
			x = 0
			y = 0
			temp = 0
			Temp = 0
			for pos in self.region_list[key]:
				temp1 = self.grid[pos[0]][pos[1]].final_temp
				if pos[0]==0 or pos[0]+1==self.rows or pos[1]==0 or pos[1]==self.cols:
					temp1 -= 1		
				Temp += temp1
				x += float(pos[0])/float(temp1)
				y += float(pos[1])/float(temp1)
				temp += float(1)/float(temp1)

			#print temp,key
			comx = int(round(x/temp))
			comy = int(round(y/temp))
			comtemp = Temp/float(len(self.region_list[key]))
			comx = max(comx,0)
			comx = min(comx,self.rows-1)
			comy = max(comy,0)
			comy = min(comy,self.cols-1)
			self.add_ac(comx,comy,comtemp)

	def get_distribution(self,row,col,curr_region):
		def check_valid(r1,c1,r2,c2):
			if r1<0 or r1>=self.rows or c1<0 or c1>=self.cols or self.regions[r1][c1] != 1e9 or self.region_size[self.regions[r2][c2]] >= self.regionmaxsize:
				return 1
			region = self.regions[r2][c2]
			temp = self.grid[r1][c1].final_temp
			tm = min(self.region_min[region],temp)
			tM = max(self.region_max[region],temp)
			if tM - tm <= self.region_cutoff or self.region_size[self.regions[r2][c2]]<self.regionminsize:
				self.regions[r1][c1] = self.regions[r2][c2]
				self.region_min[region] = tm
				self.region_max[region] = tM
				self.region_size[region] += 1
				return 2
			else:
				return 3

		q = Queue.Queue()
		self.regions[row][col] = curr_region
		q.put((row,col))
		self.region_max[curr_region] = self.grid[row][col].final_temp
		self.region_min[curr_region] = self.grid[row][col].final_temp
		self.region_size[curr_region] = 1
		while not q.empty():
			curr = q.get()
			r = curr[0]
			c = curr[1]
			flag = 0
			T = [-1,1]
			for t in T :
				f = check_valid(r+t,c,r,c) 
				if f == 2:
					q.put((r+t,c))
				elif f==3:
					flag = 1

				f = check_valid(r,c+t,r,c)
				if f == 2 :
					q.put((r,c+t))
				elif f == 3 :
					flag = 1
			if flag :
				break


	def get_surround_info(self,row,col,matrix):
		total_temp = 0
		total = 0
		neighbours = []
		if row-1>=0 :
			total_temp += matrix[row-1][col]
			total += 1
			neighbours.append((row-1,col))
		if col-1>=0:
			total_temp += matrix[row][col-1]
			total += 1
			neighbours.append((row,col-1))
		if row+1<self.rows:
			total_temp += matrix[row+1][col]
			total += 1
			neighbours.append((row+1,col))
		if col+1<self.cols:
			total_temp += matrix[row][col+1]
			total += 1
			neighbours.append((row,col+1))
		return (float(total_temp + (4-total)*(23))/float(4),neighbours)

	def print_grid(self):
		for i in range(self.rows):
			for j in range(self.cols):
				print '{} '.format(self.get_temp_grid()[i][j]),
			print "\n"
		print "\n"
	
	def start_cycle(self):
		for ac in self.ac_cells :
			ac.update_cell()
		matrix = [[self.grid[i][j].temp for j in range(self.cols)] for i in range(self.rows)]
		matrix = self.simulate_cycle(matrix)
		for i in range(self.rows):
			for j in range(self.cols):
				self.grid[i][j].set_temp(matrix[i][j])
	
	def normalize(self,matrix):
		M = 0
		m = 1e9
		for i in range(self.rows):
			for j in range(self.cols):
				M = max(M,matrix[i][j])
				m = min(m,matrix[i][j])
		if M==m :
			return [[1 for j in range(self.cols)]for i in range(self.rows)]
		return [[(matrix[i][j])/(100) for j in range(self.cols)]for i in range(self.rows)]


	def get_power_consumption(self):
		power = 0
		for cell in self.ac_cells:
			if cell.hasac:
				power += cell.power * cell.eff
		return power

	def get_temp_grid(self):
		return self.normalize([[self.grid[i][j].temp for j in range(self.cols)] for i in range(self.rows)])
	
	def get_final_temp_grid(self):
		return self.normalize([[self.grid[i][j].final_temp for j in range(self.cols)] for i in range(self.rows)])
	
	
	def get_temp_string(self):
		s = ""
		for i in range(self.rows):
			for j in range(self.cols):
				s = s + str(format(self.grid[i][j].temp, '.2f')) + "  "
			s = s + "\n\n"
		return s

	def calculate_satisfaction(self,t1,t2):
		a1 = (t1-t2)*(t1-t2)
		b1 = math.exp(-a1/30)
		return b1

	def get_satisfaction_string(self):
		s = ""
		for i in range(self.rows):
			for j in range(self.cols):
				a1 = self.calculate_satisfaction(self.grid[i][j].final_temp,self.grid[i][j].temp)
				s = s + str(format(a1, '.2f')) + "  "
			s = s + "\n\n"
		return s
	def get_total_satisfaction(self):
		s = 0
		for i in range(self.rows):
			for j in range(self.cols):
				a1 = self.calculate_satisfaction(self.grid[i][j].final_temp,self.grid[i][j].temp)
				s = s + a1
		return (s*1.0)/(self.rows*self.cols)
	
	def get_working_acs(self):
		total = 0
		for cell in self.ac_cells:
			if cell.hasac :
				total += 1
		return total

	def simulate_cycle(self,matrix):
		queue = Queue.Queue()
		vis = {}
		for ac in self.ac_cells :
			if self.grid[ac.row][ac.col].hasac :
				queue.put((ac.row,ac.col))

		while not queue.empty():
			curr = queue.get()
			if curr not in vis :
				vis[curr] = True 
				surround_temp,neighbours = self.get_surround_info(curr[0],curr[1],matrix)
				if not self.grid[curr[0]][curr[1]].hasac :
					temp_diff = self.k * (matrix[curr[0]][curr[1]] - surround_temp)
					matrix[curr[0]][curr[1]] -= temp_diff
					
				for neighbour in neighbours:
						queue.put(neighbour)
		matrix = matrix + 0.01*np.random.rand(self.rows,self.cols)
		return matrix



from Tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

root = Tk()
root.geometry("2000x2000")

left = Frame(root, borderwidth=2, relief="solid")
right = Frame(root, borderwidth=2, relief="solid")
label3_text = StringVar()
label4_text = StringVar()
label5_text = StringVar()
label6_text = StringVar()
label7_text = StringVar()
label8_text = StringVar()

grid = ''

fn = 'CPvsTvrZ0Base.csv'
# row = ['time','Power','Cummulative Power','saticfaction']
# row = ['saticfaction']
# with open(fn, 'a') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerow(row)
# csvFile.close()

def run():
	
	# with open("input.txt") as f :
	# 	for i,line in enumerate(f) :
	# 		if i == 0 :
	# 			a , b = line.split()
	# 			grid = GRID(int(a),int(b))
	# 		else :
	# 			a , b , c = line.split()
	# 			grid.set_final_temp(int(a),int(b),float(c))
	
	input = np.loadtxt("inputPrint.txt", dtype='i', delimiter=',')
	# input = np.random.randint(15,19, size=(10, 10))

	grid = GRID(input.shape[0],input.shape[1])
	for i in range(input.shape[0]):
		for j in range(input.shape[1]):
			grid.set_final_temp(int(i),int(j),float(input[i][j]))

	for i in range(grid.rows) :
		for j in range(grid.cols) :
			print '{} '.format(grid.grid[i][j].final_temp),
		print "\n"
	
	grid.start_distribution()

	print "Optimum Number Of Acs : " , len((grid.ac_cells))
	print "Optimum Positions For Acs : "
	for ac in grid.ac_cells :
		print "Region : " , grid.regions[ac.row][ac.col] , " Position : " , ac.row , ac.col

	
	for i in range(grid.rows) :
		for j in range(grid.cols) :
			print '{} '.format(grid.regions[i][j]),
		print "\n"
	

	fig = plt.figure()
	ax = fig.add_subplot(111)
	im = ax.imshow(np.array(grid.get_temp_grid()), cmap='plasma', interpolation='nearest')
	canvas = FigureCanvasTkAgg(fig, master=left)
	canvas.show()
	canvas.get_tk_widget().pack()
	TS = 0
	for i in range(3600):
		#time.sleep(0.1)
		grid.start_cycle()
		#grid.print_grid()

		im = ax.imshow(np.array(grid.get_temp_grid()) , cmap='plasma', interpolation='nearest')
		P = grid.get_power_consumption() 
		# print P
		grid.total_power += P
	
		S = grid.get_total_satisfaction()
		TS = TS + S
		print S

		row =  [S]
		with open(fn, 'a') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerow(row)
		csvFile.close()

		label3_text.set("Power Consumption : \n"+ str(P))
		label4_text.set("\nTemperature Grid : \n" + grid.get_temp_string())
		label5_text.set("\nWorking ACS : \n" + str(grid.get_working_acs()))
		label6_text.set("\nTotalPower:\n" + str(grid.total_power))
		label7_text.set("\nsatisfaction:\n" + grid.get_satisfaction_string())
		label8_text.set("\nTotal satisfaction:\n" + str(grid.get_total_satisfaction()))
		# canvas.draw()
	print ("AC ALL ARE ON",grid.region_count*5507*3.5*3600)#416329200
	print ("AC Total power consumtion",grid.total_power)
	print ("No Of AC",grid.region_count)
	print ("Percent of power gain",(((grid.region_count*5507*3.5*3600)-grid.total_power)/(grid.region_count*5507*3.5*3600))*100)
	print (TS*1.0/3600)

label1 = Label(left, text="Room Temperature Simulation")
label2 = Button(left, text="Run Simulation",command=run)
label3 = Label(right, textvariable=label3_text)
label4 = Label(right,textvariable=label4_text)
label5 = Label(right,textvariable=label5_text)
label6 = Label(right,textvariable=label6_text)
label7 = Label(right,textvariable=label7_text)
label8 = Label(right,textvariable=label8_text)
left.pack(side="left", expand=True, fill="both")
right.pack(side="right", expand=True, fill="both")
label1.pack()
label2.pack()
label3.pack()
label4.pack()
label5.pack()
label6.pack()
label7.pack()
label8.pack()
root.mainloop()
while(1):
	run()	
	break
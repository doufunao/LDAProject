import pickle
import random
import os
 
class Lda:
	def __init__(self,vocabulary,words,topicNums=10,beta=0.01,iteration=50,saveStep=10,beginSaveIters=10):
		self.vocabulary = vocabulary
		self.words = words
		self.topicNums=topicNums
		self.alpha=topicNums/50
		self.beta=beta
		self.iteration=iteration
		self.saveStep=saveStep
		self.beginSaveIters=beginSaveIters

	def initialize(self):
		self.M = len(self.words)
		self.K = self.topicNums
		self.V = len(self.vocabulary)
		self.z = [[random.randrange(self.K) for j in range(len(self.words[i]))]for i in range(self.M)]
		self.w = [[self.vocabulary.index(word) for word in document_words] for document_words in self.words]
		self.nmk = [[0 for j in range(self.K)] for i in range(self.M)]
		self.nkt = [[0 for j in range(self.V)] for i in range(self.K)]
		self.nm = [0 for i in range(self.M)]
		self.nk = [0 for i in range(self.K)]
		self.theta = [[0 for j in range(self.K)] for i in range(self.M)]
		self.phi = [[0 for j in range(self.V)] for i in range(self.K)]

		#initialisation
		[[self.__work4Initalize(m,n,self.z[m][n]) for n in range(len(self.z[m]))]for m in range(len(self.z))]

	def __work4Initalize(self,m,n,k):
		t = self.w[m][n]
		self.nmk[m][k] = self.nmk[m][k]+1
		self.nkt[k][t] = self.nkt[k][t]+1
		self.nm[m] = self.nm[m]+1
		self.nk[k] = self.nk[k]+1

	def inferenceModel(self):
		for i in range(self.iteration):
			# print("iteration : ",i)
			if i>self.beginSaveIters and i%self.saveStep ==0:
				self.updateEstimateParameters()


			for m in range(len(self.z)):
				for n in range(len(self.z[m])):
					newtopic = self.gibbsSampling(m,n)
					self.z[m][n] = newtopic 

		self.saveModel(20)

	def gibbsSampling(self,m,n):
		kold = self.z[m][n]
		t = self.w[m][n]
		self.nmk[m][kold] = self.nmk[m][kold]-1
		self.nm[m] = self.nm[m] -1
		self.nkt[kold][t] = self.nkt[kold][t]-1
		self.nk[kold] = self.nk[kold]-1

		p_k = [self.__compute_pk(m,k,t) for k in range(self.K)]
		
		for i in range(1,len(p_k)):
			p_k[i] = p_k[i]+p_k[i-1]


		u = random.random() * p_k[len(p_k)-1]
		newtopic = -1
		for newtopic,p in enumerate(p_k):
			if p > u: break 

		self.nmk[m][newtopic] = self.nmk[m][newtopic]+1
		self.nkt[newtopic][t] = self.nkt[newtopic][t]+1
		self.nm[m] = self.nm[m]+1
		self.nk[newtopic] = self.nk[newtopic]+1

		return newtopic

	# this is p(zi=k|vector_z_without_zi,vector_w) = (nmk[m][k] + alpha) / (nm[m] + K * alpha) * (nkt[k][t] + beta) / (nk[k] + V * beta);
	def __compute_pk(self,m,k,t):
		return (self.nmk[m][k] + self.alpha) / (self.nm[m] + self.K * self.alpha) * (self.nkt[k][t] + self.beta) / (self.nk[k] + self.V * self.beta)


	def updateEstimateParameters(self):
		# update theta
		[[self.__update_theta_mk(m,k) for k in range(self.K)]for m in range(self.M)]
		# update phi
		[[self.__update_phi_kt(k,t)   for t in range(self.V)]for k in range(self.K)]


	def __update_theta_mk(self,m,k):
		self.theta[m][k] = (self.nmk[m][k] + self.alpha) / (self.nm[m] + self.K * self.alpha)

	def __update_phi_kt(self,k,t):
		self.phi[k][t]  =  (self.nkt[k][t] + self.beta) / (self.nk[k] + self.V * self.beta)


	def saveModel(self,topN):
		if not os.path.exists("result"):
			os.mkdir("result")
		
		with open("./wordResult.out","w") as fw:
			for k in range(self.K):
				phi_2_vocu = [v for v in range(self.V)]
				phi_2_vocu.sort(key=lambda x:self.phi[k][x],reverse=True)
				fw.write("topic "+str(k)+"  :  ")
				for n in range(topN):
					fw.write(self.vocabulary[phi_2_vocu[n]]+":"+str(self.phi[k][phi_2_vocu[n]])+"  ")
				fw.write("\n")

		with open("./phi.out","w") as fw:
			for k in range(self.K):
				phi_2_vocu = [v for v in range(self.V)]
				phi_2_vocu.sort(key=lambda x:self.phi[k][x],reverse=True)
				fw.write("topic "+str(k)+" :   ")
				for v in range(self.V):
					fw.write(self.vocabulary[phi_2_vocu[v]]+":"+str(self.phi[k][phi_2_vocu[v]])+"   ")
				fw.write("\n")

		with open("./theta.out","w") as fw:
			for m in range(self.M):
				fw.write("document "+str(m)+" :   ")
				theta_index =[j for j in range(self.K)]
				theta_index.sort(key=lambda x:self.theta[m][x],reverse=True)
				for k in range(self.K):
					fw.write("topic "+str(theta_index[k])+":"+str(self.theta[m][theta_index[k]])+"   ")
				fw.write("\n")

		with open("./topic_matrix.out","w") as fw:
			for m in range(self.M):
				# fw.write("document "+str(m)+" :   ")
				theta_index =[j for j in range(self.K)]
				theta_index.sort(key=lambda x:self.theta[m][x],reverse=True)
				theta_str = [str(x) for x in self.theta[m]]
				fw.write(','.join(theta_str))
				# for k in range(self.K):
					# fw.write(str(self.theta[m][theta_index[k]]))
				fw.write("\n")

def test():
	# obj = Lda(,topicNums=300)
	obj.initialize()
	print("len(obj.M): ",obj.M)
	print("len(obj.K): ",obj.K)
	print("len(obj.V): ",obj.V)
	print("len(obj.z): ",len(obj.z))
	print("len(obj.z[0]): ",len(obj.z[0]))
	print("len(obj.nmk): ",len(obj.nmk))
	print("len(obj.nmk[0]): ",len(obj.nmk[0]))
	print("z[0].count(1): ",obj.z[0].count(1))
	print("nmk[0][1]: ",obj.nmk[0][1])
	obj.inferenceModel()

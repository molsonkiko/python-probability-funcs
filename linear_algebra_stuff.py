#!python3
'''
Library of linear algebra algorithms made for the Linear Algebra- Foundations to Frontiers
(LAFF) course from EdX.
MORE NOTES BELOW THE FUNCTIONS

DEPENDENCIES INSTALLABLE BY pip: scipy, numpy
DEPENDENCIES FROM MY MODULES: prob_calc

MAJOR FUNCTIONS IN THIS MODULE:

1. gaussElim: Does Gaussian elimination on a matrix, and returns its reduced-row-echelon form along with its determinant, and the vector that solves the system of linear equations when you specify a solution vector (solnvec). You can make the function "show its work" by setting the showSteps option to True. Can also calculate the inverse of a matrix. Probably my most used function in this module.
2. gramschmidt: Orthogonalizes (optionally orthonormalizes) the columns of a matrix. Can also return the QR factorization of the matrix (which is strictly inferior to np.linalg.qr(mat)).
3. llsq: Performs linear least-squares on a system of equations (that is, a matrix with one column of ones and one or more columns of independent variables, and a vector of same length as the number of rows of the matrix). Strictly inferior to np.polyfit(x,y) followed by np.polyval(fit,x).
4. eigenDecompose: A convenience function that gets the eigenvalues and eigenvectors of a matrix and returns the eigendecomposition of the matrix (the matrix of eigenvectors, the matrix of eigenvalues, and the inverse of the matrix of eigenvectors).
5. eigenvectorsFromEigenvalues: Given the eigenvalues of a matrix, finds the corresponding eigenvectors by finding the bases of the nullspace of (A-lam*I), where A is the matrix and lam is an eigenvalue.
6. powerMethod: An implemenation of the power method to find the largest eigenvalue of a matrix (and maybe the smallest too, if the matrix is invertible).
7. det_recursive: Remember the horrible, terrible, awful recursive determinant algorithm that your linear algebra teachers tormented you with? Yeah, this does that. Please don't try it for a matrix larger than 9x9.

And last but not least, my personal favorite:
8. characteristic_polynom: Finds the characteristic polynomial of a matrix, and, if you so desire, its eigenvalues (the roots of the characteristic polynomial) and the eigenvectors (found by eigenvectorsFromEigenvalues). This has absolutely no practical value, but it uses my polynom class and it is just nifty. Because it uses det_recursive, it should never ever ever be used on a matrix larger than 9x9.
'''
#A handy shortcut: If M and v are np.arrays, M @ v = M.dot(v). This is true in general.
#Also, np.allclose(v,w,atol=tolerance) returns True if all the entries of v are within tolerance of the
#	corresponding entries of w.
import math, time, re
import numpy as np
from prob_calc import polynom
from phys_constants import *
import logging
logging.basicConfig(level=logging.ERROR)
from scipy.linalg import null_space


def gaussElim(mat,solnvec=None,showSteps=False, calcInverse=False):
	'''
	mat: an n*n matrix.
	solnvec: a 1*n vector, the transpose of the column vector
	representing the values the linear equations in the system.
	symmPosDef should be set to true if you're dealing with a symmetric positive
	definite matrix
	If not calcInverse, Returns: A dictionary, {'det': the determinant of the matrix,
	'solnvec': the solution vector, 'GauXF': the Gauss transform matrix, 
	'UTmat': the upper triangularized original matrix}.
	Example:
	mat=[[1,2],
		 [2,3]]
	solnvec=[3,4]
	[1 2|3]
	[2 3|4]
	-->
	[1 2|3]
	[0 -1|-2]
	So x1=2. Thus,x0=-1.
	So gaussElim will return {'solnvec':[-1,2],'GauXF':[[1,0],[-2,1]], 'UTmat':[[1,2],[0,1]]}
	If calcInverse, performs Gauss-Jordan Elimination to diagonalize the Gauss-eliminated
	matrix and returns {'solnvec':[-1,2],'inverse':[[-3,2],[2,-1]]}
	'''
	#THIS IS WEEK 8 IN THE LAFF COURSE
	#matcopy=np.array(mat.copy(),'float')
	mat=np.array(mat,'float')
	r=len(mat)
	c=len(mat[0])
	det=1 #the determinant will initally be set to 1.
	if solnvec is None:
		solnvec=np.zeros([1,r])[0]
	solnvec=np.array(solnvec,'float')
	v=len(solnvec)
	if r!=v:
		return "Vector dimension does not equal number of rows of matrix."
	
	transformMat=np.eye(r)

	if showSteps:
		print("Start")
		print(mat,solnvec)

	for s in range(r):
		xformThisStep=np.eye(r)
		if showSteps:
			print("\nStep",str(s+1))
		if mat[s][s]==0:
			#Check if there are any zeros on the diagonal.
			#If a zero is found, find the first nonzero entry directly
			#below it and swap that nonzero entry's row with the current row.
			pivotMat=np.eye(r)
			nonZeroEntryFound=False
			for i in range(s+1,r):
				if mat[i][s]>5e-15 or mat[i][s]<-5e-15: #Python frequently produces numbers like 1.1234938423e-16 that should definitely be zero but aren't because floating point arithmetic
					pivotMat[s][s]=pivotMat[i][i]=0
					pivotMat[s][i]=pivotMat[i][s]=1
					nonZeroEntryFound=True
					mat=pivotMat.dot(mat)
					transformMat=pivotMat.dot(transformMat)
					det = det * -1 #swapping rows flips the sign of the determinant.
					if showSteps:
						print("Swapped row",str(s+1),"and row",str(i+1)+".")
					break
			if not nonZeroEntryFound: #Gaussian elimination fails- get bases of null space.
				#There's no unique solution if you can't remove a zero on the diagonal
				#or if the matrix is not square.
				return solveSingularMatrix(mat,solnvec,transformMat,showSteps)

		for i in range(r): #Zero out all entries directly below the current entry.
			if i==s:
				continue #Don't operate on entries on the diagonal.
			if i<s and not calcInverse:
				continue #Only operate on entries above the diagonal if you want to
				#calculate the inverse.
			mult=mat[i][s]/mat[s][s]
			xformThisStep[i][s]=-mult #This zeros out the sth entry in column i (directly
			#below the focal cell).
			#solnvec[i]-=solnvec[s]*mult
			if showSteps and mult!=0:
				print("Row",str(i+1),"of matrix -= row",str(s+1),"*",mult)
		mat=xformThisStep.dot(mat) #Zero out all entries directly below mat[s][s]
		logging.info("xformThisStep=%s",(str(xformThisStep),))
		transformMat=xformThisStep.dot(transformMat)
		logging.info("transformation matrix=%s",(str(transformMat),))
		if showSteps:
			for i in range(r):
				print("Row",str(i+1)+": ",list(mat[i]))
			#print("transform matrix=")
			#print(transformMat)

	if r!=c:
		return solveSingularMatrix(mat,solnvec,transformMat,showSteps)
	
	for i in range(r):
		det *= mat[i,i] #now we calculate the determinant of the upper triangular matrix.
	
	if calcInverse:
		diagMultMat=np.eye(r)
		for i in range(r): #Normalize all the entries on the main diagonal
			diagMultMat[i][i]/=mat[i][i]
		inverse=diagMultMat.dot(transformMat)
		solnvec=inverse.dot(solnvec)
		return {'det':det,'solnvec':solnvec, 'inverse': inverse}

	else:
		#Now that the matrix is in upper triangular form, we solve the system of
		#linear equations. This is called back substitution.
		solnvec=backSub(mat,transformMat,solnvec,showSteps)
		return {'det':det,'solnvec':solnvec,'GauXF':transformMat, 'UTmat':mat}

def backSub(utmat,xformMat,solnvec,showSteps=False):
	'''
	Given the upper triangular matrix (utmat) and the Gauss transform matrix (xformMat)
	obtained from Gaussian Elimination, and some vector that represents the set of
	solutions for the linear equations in utmat and xformMat, solve the system of linear
	equations.
	Returns: a vector, the values for each of the parameters, chi0 through chiN.
	'''
	r=len(utmat)
	#Apply the same series of transformations to the solution vector that we
	#applied to the original matrix.
	solnvec=xformMat.dot(solnvec) #This step is known as forward substitution
	GJelimMat=np.eye(r) #This will become the matrix that transforms the upper
	#triangular matrix utmat into the identity matrix.
	
	if showSteps:
		print("Solution vector after Gaussian elimination\n(but before solving system of linear equations) = "+str(solnvec))
	solnvec[r-1]=solnvec[r-1]/utmat[r-1][r-1]
	
	#Now we solve the system of linear equations from bottom to top, substituting in
	#the values of the parameters as we calculate them.
	for i in range(2,r+1):
		# newXformMat=np.eye(r)
		for x in range(1,i):
			solnvec[r-i]-=utmat[r-i][r-x]*solnvec[r-x]
		solnvec[r-i]/=utmat[r-i][r-i]
			# newXformMat[r-x][r-i]=-utmat[r-x][r-i]/utmat[r-i][r-i]
		# GJelimMat=GJelimMat.dot(newXformMat)
		# print(GJelimMat)
		
	return solnvec

def eigenDecompose(mat):
	'''Returns Q, DEL, and Q^-1 from the eigendecomposition of square matrix mat where
Q*DEL*Q^-1 = A, where Q has the eigenvectors of mat as its columns and 
	DEL is a diagonal matrix with the corresponding eigenvalues on its main diagonal.'''
	mval,mvec=np.linalg.eig(mat)
	mval=np.diag(mval)
	mvi=np.linalg.inv(mvec)
	return mvec,mval,mvi

def solveSingularMatrix(mat,solnvec,transformMat,showSteps=False):
	'''
	TODO:
	1. Get this to work properly for a non-square matrix. 
	
	Returns the basis for the null space of a singular (non-invertible) matrix (mat).
	Also determines if the given vector (solnvec) is in the column space of mat.
	'''
	r=len(mat)
	c=len(mat[0])
	solnvec=transformMat.dot(solnvec)
	if showSteps:
		print(mat)
		print(solnvec)
		print("This system of linear equations has no unique solution.")
		print("Returning the bases for the null space of the matrix.")
	freeVariables=dict() #Find all the free variables in the system.
	nonFreeVariables=dict()
	
	nullbases={'validVec':True} #Create a dict that maps each free variable to the
	#corresponding basis vector for the null space.
	#The validVec property of nullbases indicates whether solnvec is in the column space of
	#the matrix mat.
	
	for row in range(r):
		if row>c:
			break
		if mat[row][row]==0:
			for col in range(row,c):
				if col not in {**freeVariables,**nonFreeVariables}:
					if mat[row][col]==0:
						freeVariables[col]=row
					else:
						nonFreeVariables[col]=row
						break
		else:
			nonFreeVariables[row]=row
	
	noSolution=False
	freevarList='free variables: '
	for col in freeVariables:
		freevarList+="chi"+str(col)+", "
		if solnvec[col]!=0:
			noSolution=True
	if showSteps:
		print(freevarList[:-2])
	if noSolution:
		nullbases['validVec']=False
		if showSteps:
			print("There is no solution at all for this system of equations, because a free variable is set to nonzero.")
		
		
	#Get the matrix as close to reduced-row-echelon form as possible.
	#Each row should only have one non-free variable in it.
	for pivCol in nonFreeVariables:
		pivRow=nonFreeVariables[pivCol]
		for row in range(pivRow):
			mult=mat[row][pivCol]/mat[pivRow][pivCol]
			mat[row] -= mult*mat[pivRow]
	
	for fvar in freeVariables:
		nullbase=np.zeros([1,c])[0]
		nullbase[fvar]=1 #Each basis vector for the null space has one free variable
		#set to one and the rest set to zero.	 
		for z in range(r):
			row=mat[z]
			if row[fvar] == 0: 
				continue
			#Find all non-free variables in the same row as a nonzero value for 
			#the free variable, and find a value for that non-free variable such that
			#the row sums up to zero.
			for nvar in range(c):
				if nvar==fvar or row[nvar]==0 or nvar in freeVariables:
					continue
				nullbase[nvar] = -row[fvar]/row[nvar]
		nullbases['basis'+str(fvar)]=nullbase
	
	if showSteps:
		print("Returning the bases for the null space of the matrix.")
	return nullbases

def llsq(eqns,y,verbose=False):
	'''Solves a linear least-squares problem.
Given an overdetermined system of equations (an m*n matrix eqns, where m>n) and y, an
m-vector corresponding to the value of the dependent variable associated with the 
independent variables in the matrix, returns a tuple (x,yap)
x=the set of parameters for eqns such that eqns(x)=yap. 
yap=the projection of y into the column space of eqns and the least-squares solution of the system.
	'''
	y=np.array(y)
	eqns=np.array(eqns)
	r=len(y)
	at=eqns.transpose()
	ata=at.dot(eqns) 
	atai=np.linalg.inv(ata)
	aty=at.dot(y)
	x=atai.dot(aty)
	yap=eqns.dot(x)
	if verbose:
		print('eqns transpose dot y')
		print(aty)
		print('eqns transpose dot eqns')
		print(ata)
		print('(eqns transpose dot eqns)^-1')
		print(atai)
		print("left pseudo-inverse of eqns=(eqns transpose dot eqns)^-1 dot eqns transpose")
		print(atai.dot(at))
		print("x = pseudo-inverse dot y = the parameters of the best-fit equation.")
		print("yap = eqns(x) = the projection of y into the column space of eqns.")
		print('''yap[col] also equals the value of the best fit equation for the values of the
independent variables in column col.''')
	return (x,yap)

def gramschmidt(mat,normalize=True,get_QR=False,verbose=False):
	'''Uses the Gram-Schmidt process to transform the columns of a matrix
into an orthogonal set. Optionally normalize all the columns as well.
	Normalization is required for the QR factorization.
If get_QR is True, does QR factorization and returns a dict:
	{'Q': the orthonormal result of the gram-schmidt process, 'R': the result of Q^T dot A, 
	the other part of the QR factorization}.
If verbose, describes the operations used to orthogonalize the matrix.

USE np.linalg.qr INSEAD OF THIS FOR QR FACTORIZATION!!! 
THIS MAY NOT GET THE CORRECT QR FACTORIZATION OF SOME MATRICES (probably just singular ones)!!!'''
	if get_QR and not normalize:
		logging.warn("The QR factorization cannot pe performed without normalizing the orthogonal matrix.")
		normalize=True
	ncols=len(mat[0])
	r=len(mat)
	badcols=[]
	mat=np.array(mat,float)
	if get_QR:
		mat_copy = np.array(mat[:,:],float)
	if verbose:
		print(mat)
	for col2 in range(1,ncols):
		c2=mat[:,col2]
		for col in range(col2):
			if col in badcols:
				continue #col has all zeros- would get zerodiv error
			c=mat[:,col]
			cT=c.transpose()
			#((c.transpose().dot(c))^-1).dot(c.transpose()) is known as the left pseudo-inverse
			#of c, because c.dot(leftpseudoinv(c)) = I, but leftpseudoinv(c).dot(c)!=I in general.
			if verbose:
				print("col",str(col2),"-= "+str(cT.dot(c2)/cT.dot(c))+"*col",str(col))
			c2 -= c*cT.dot(c2)/cT.dot(c)
			zcount=0
			for i in range(r):
				if c2[i]<=5e-15 and c2[i]>=-5e-15:
					zcount+=1
			if zcount==r:
				badcols.append(col2)
				if verbose:
					print("Columns "+str(col2)+" and "+str(col)+" are linearly dependent.")
		if verbose:
			print(mat)
	if normalize:
		for col in range(ncols):
			if col in badcols:
				continue
			c=mat[:,col]
			norm=(c.dot(c))**0.5
			c/=norm
			if verbose:
				print("Normalizing column",str(col),"by dividing by "+str(norm))
	
	if get_QR:
		R = mat.T @ mat_copy
		for row in range(R.shape[0]):
			for col in range(row):
				R[row,col]=0 #R is an upper triangular matrix, but unless we zero those entries
							#out, they will be shown as numbers very close to but not exactly 0.
		return {'Q': mat, 'R': R}
	
	return mat

def qr_eigvals(mat,maxloops=2000):
	'''An algorithm for finding the eigenvalues of SOME matrices by copying the matrix,
and then repeatedly calculating the QR factorization of the copy and overwriting it
with R @ Q (as opposed to Q @ R, which is just the copy). Eventually copy will have the
eigenvalues of mat on its diagonal.
Returns the eigenvalues of mat.'''
	loops=0
	cop=mat.copy()
	while True:
		loops+=1
		if np.allclose(cop,np.triu(cop)):
			return np.diag(cop)
		if np.allclose(cop,np.tril(cop)):
			return np.diag(cop)
		qr=lin.gramschmidt(cop,True,True)
		cop=qr['R']@qr['Q']
		if loops==maxloops:
			break

def eigenvectorsFromEigenvalues(mat,vals):
	'''For each val in vals, finds the bases of the null space of the matrix mat - val*I.
Since an eigenvector v of a matrix A with eigval lam is defined by the relation (A-lam*I) @ v = 0,
the bases of the nullspace of mat - val*I are the eigenvectors of A corresponding to those values.
Returns: a matrix with the eigenvectors corresponding to the values in vals as columns.'''
	try:
		b=int(vals)
		vals=[vals]
	except: #vals is not an numeric datatype
		pass
	#the corresponding vectors are the bases of the null space of the matrix mat - val*I,
	#because (mat - val*I) @ vec = 0, since mat@vec = val*vec and val*I@vec = val*vec also.
	vecs=[null_space(mat-val*np.eye(mat.shape[0])) for val in vals]
	#lin.gaussElim should in theory get the bases of the null space of a singular matrix, but
	#in practice it is numerically unstable so it won't do the job right.
	for i in range(len(vecs)):
		vecs[i] /= np.linalg.norm(vecs[i])
	return np.column_stack(vecs)

def powerMethod(mat,max_loops=1000,get_min_also=False,abs_tolerance=1e-6,verbose=False):
	'''Determines the dominant eigenvalue of matrix mat by generating a random vector and repeating the process
of multiplying the matrix by the vector and normalizing the vector until the vector
	is elementwise identical (within abs_tolerance) to the vector on the previous iteration.
	This method finds the largest eigenvalue of the matrix and its associated eigenvector.
max_loops: The maximum number of iterations that the function will do before it gives up.
If get_min_also is true, this function will try to invert mat and (assuming mat is invertible)
	use the power method to find the largest eigenvector of mat^-1, which is the smallest eigenvector
	of mat.
If verbose, prints a description of the last step on each loop.'''
	vec=np.random.random(size=mat.shape[1])
	if get_min_also:
		#If mat is invertible, and L is an eigenvalue of mat with corresponding eigenvector V,
		#then 1/L is an eigenvalue of mat^-1, also with corresponding eigenvector V.
		try:
			minval,minvec = powerMethod(np.linalg.inv(mat),max_loops,False,verbose)
		except:
			print("This matrix is not invertible, so the power method cannot be used to get the smallest eigenvalue and its corresponding eigenvector.")
	if verbose:
		print("Starting with randomly generated vector "+str(vec))
	loops=0
	while True:
		loops+=1
		last=vec[:]
		vec=mat@vec
		new_mult = (vec@vec)**0.5
		vec /= new_mult
		if verbose:
			print("Loop {0}: Multiplying vector by matrix and then dividing by {1} to normalize. Vector is now {2}".format(loops,new_mult,vec))
		if np.allclose(vec,last,atol=abs_tolerance):
			break
		if loops==max_loops:
			print("Ending iteration after "+str(max_loops)+" iterations because convergence is unlikely.")
			break
	mulvec = mat @ vec
	mul=[]
	for i in range(len(mulvec)):
		if not abs(vec[i])<=1e-16:
			mul.append(mulvec[i]/vec[i])
	eigval = np.mean(mul)
	if verbose:
		print("Converged to a dominant eigenvalue of {0} with associated eigenvector {1}.".format(eigval,vec))
	if get_min_also:
		return {'max':(eigval,vec), 'min': (1/minval,minvec)}
	
	return eigval,vec

def is_triangular(mat):
	'''Checks if matrix is triangular or diagonal.'''
	tests = [
		np.allclose(mat, np.tril(mat)), # check if lower triangular
		np.allclose(mat, np.triu(mat)), # check if upper triangular
		np.allclose(mat, np.diag(np.diag(mat))) # check if diagonal
		]
	return any(tests)

def det_recursive(mat):
	'''Implements the super-lame recursive method for calculating the determinant,
where you have to calculate the determinants of n!/2 2x2 matrices to get the determinant of a matrix.
Seriously, just use my gaussElim function to get the determinant ('det' in the returned dict).'''
	shape=mat.shape
	if shape[0]!=shape[1]:
		raise ValueError("This function only works for square matrices.")
	if shape==(2,2):
		return mat[0,0]*mat[1,1]-mat[0,1]*mat[1,0]
	out=0
	row=mat[0,:]
	for i in range(shape[0]):
		submat=np.column_stack((mat[1:,:i],mat[1:,i+1:]))
		if i%2==0:
			out+=row[i]*det_recursive(submat)
		else:
			out -= row[i]*det_recursive(submat)
	return out

def characteristic_polynom(mat,get_eigens=False):
	'''Given a matrix, mat, calculates its characteristic polynomial as an np.poly1d object.
mat is an np.ndarray, a list of lists, or tuple of lists.
You can get the eigenvalues of mat (which are the roots of this polynomial)
by looking at the "roots" attribute (not a method, an attribute) of the return value of this function.
If get_eigens is True, returns a tuple (characteristic polynomial, eigenvalues, matrix of eigenvectors).
Otherwise, returns just the characteristic polynomial.
For the love of God, don't try using this on a matrix that's 10x10 or larger. You will be waiting all day.

Determines the characteristic polynomial by this process:
1. Convert all columns of the matrix into my polynom data type,
where the scalars are zero-degree polynomials and the i^th diagonal entry is A^ii - 1*x.
3. Calculate the determinant of this matrix.
The polynom data type supports multiplication and addition of polynoms.
4. This will return an nth-degree polynomial with roots equal to the eigenvalues of the matrix.

THIS MAY NOT ACTUALLY GET ALL OF THE EIGENVECTORS. IN FACT, IT MAY GET NONE AT ALL.
This is because the eigenvalues calculated by this function are not exact, so
mat - val*I may NOT be singular for a val calculated by this function, which means that
scipy.null_space (which is what's actually finding the eigenvectors from the eigenvalues),
returns nothing, instead of returning the eigenvector(s) corresponding to val, as
it would do if val were exact.
Also, IF THE MATRIX HAS MULTIPLE OF THE SAME EIGENVALUE, IT MAY RETURN REDUNDANT COPIES OF CORRESPONDING
EIGENVECTORS. Not sure.
	'''
	if type(mat) == np.ndarray:
		mat = [list(x) for x in mat]
	cop = np.array(mat.copy())
	r,c= len(mat),len(mat[0])
	if r!=c:
		raise ValueError("Can only calculate the characteristic_polynom of a square matrix.")
	for i in range(r):
		for j in range(c):
			if i==j:
				mat[i][j] = [-1,mat[i][j]]
			mat[i][j]=polynom(mat[i][j])
			#each diagonal entry in the matrix used to calculate the characteristic polynomial of A
			#is that entry of A minus lambda.
	mat = np.array(mat,dtype=np.object_)
	char_poly = det_recursive(mat) #the characteristic polynomial is the determinant of A - lam*I
	char_poly = np.poly1d(char_poly.args)
	if get_eigens:
		vals = sorted(char_poly.roots,key=abs,reverse=True)
		#abs not only flips negatives, it also multiplies complex numbers by their
		#complex conjugates. This is how np.linalg.eig sorts eigenvalues.
		vecs = eigenvectorsFromEigenvalues(cop,vals)
		return char_poly, vals, vecs
	return char_poly

def goodentm(testAsStr, array):
	'''
testAsStr: a string encoding a conditional test of an appropriate variable name
(e.g., "a>3", "i<=-0.3", "GX+2=15"), or no variable name at all.
array: an array, at least 2D.
Returns: a copy of the original array, with zeros wherever the condition is not met.'''
	varNm=re.match("[a-zA-Z]\w*",testAsStr)
	try:
		var=testAsStr[varNm.start():varNm.end()]
		cond=testAsStr[varNm.end():]
	except:
		cond=testAsStr
	row=len(array)
	col=len(array[0])
	out=np.zeros([row,col])
	for r in range(row):
		for c in range(col):
			i=m2[r,c]
			out[r,c]=i if eval("i"+cond) else 0
	return out
	
def goodent1(iterable, test):
	'''
test: a string encoding a conditional test (e.g., ">3", "<=-0.3", "+2==15")
array: a list or tuple (1-dimensional)
Returns: a copy of the original array, with zeros wherever the test condition is not met.'''
	out=[]
	for i in iterable:
		ent=(i if eval("i"+test) else 0)
		out.append(ent)
	return out

def partition(mat,part_size):
	'''
	Breaks the matrix into part_size * part_size submatrices. When there aren't enough
	columns or rows left to make a full part_size * part_size submatrix, uses the most
	columns or rows it can to make submatrices.
	The return value is a dictionary, where each key is the tuple (start_row,start_col)
	and each value is the submatrix for which submatrix[0][0]=mat[start_row][start_col].
	For example, suppose the following: part_size=3, 
	mat=np.array([
	   [ 2,	 0,	 1,	 2],
	   [-2, -1,	 1, -1],
	   [ 4, -1,	 5,	 4],
	   [-4,	 1, -3, -8]])
	The output would be 
	   {(0, 0): array([[ 2.,  0.,	1.],
	   [-2., -1.,  1.],
	   [ 4., -1.,  5.]]), 
	   (3, 0): array([[-4.,	1., -3.]]), 
	   (0, 3): array([[ 2.],
	   [-1.],
	   [ 4.]]), 
	   (3, 3): array([[-8.]])}
	'''
	r=len(mat)
	c=len(mat[0])
	if part_size>r or part_size>c:
		return "The partition size can't be larger than either dimension of the matrix"

	ps=part_size
	row=col=0
	xdim=ydim=part_size
	matDict=dict()
	matInProgress=False
	while row<r:
		if not matInProgress:
			ydim=min(part_size,r-row)
			xdim=min(part_size,c-col)
			matInProgress=True
			rowStart=row
			colStart=col
			newmat=np.zeros([ydim,xdim])
			colFinal=colStart+xdim
			rowFinal=rowStart+ydim
		while col<colFinal:
			newmat[row-rowStart][col-colStart]=mat[row][col]
			col+=1
		row+=1
		if row<rowFinal:
			col=colStart
		if row==rowFinal and col==colFinal:
			matInProgress=False
			#key='r'+str(rowStart)+'c'+str(colStart)
			key=(rowStart,colStart)
			matDict[key]=newmat
			if row<r:
				col=colStart
			else:
				if col>=c:
					break
				row=0
	return matDict

test_mats=[
	np.array([
	[2.92,0.86,-1.15],
	[0.86,6.51,3.32],
	[-1.15,3.32,4.57]]),
	np.array([
	[0,0,1,0],
	[1,1,2,3],
	[2,2,5,10],
	[0,0,0,0]]),
	np.array([
	[1,2,3],
	[1,2,-3],
	[2,4,7]]),
	np.array([
	[1,3,-2],
	[2,4,8]]),
	np.array([
	[1,2],
	[3,4],
	[-2,8]]),
	np.array([
	[0.4,0.3,0.1],
	[0.4,0.3,0.6],
	[0.2,0.4,0.3]]),
	np.array([[ 2,	0,	1,	2],
	   [-2, -1,	 1, -1],
	   [ 4, -1,	 5,	 4],
	   [-4,	 1, -3, -8]]),
	np.array([
		[1,4,3,9],
		[0,2,-4,8],
		[0,0,14,-20],
		[0,0,0,12]])
]
test_mats.append(np.random.normal(size=25).reshape((5,5)))
test_mats.append(np.random.normal(size=49).reshape((7,7)))
test_mats[-1][3]=test_mats[-1][6]*2.5 #make the matrix singular
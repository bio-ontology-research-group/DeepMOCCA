import sys
f=open(sys.argv[1])
lines=f.readlines()
f.close()
f=open(sys.argv[1]+'.out','w+')
for l in lines[1:]:
	index=l.find('ENSG')
	index2=l.rfind('\t')
	f.write(l[index:l.find('|',index)])
	f.write('\t')
	f.write(l[index2+1:])


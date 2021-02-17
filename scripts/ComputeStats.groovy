@Grab(group='colt', module='colt', version='1.2.0')
import cern.colt.list.*
import cern.jet.stat.Descriptive
import cern.jet.stat.Probability

def ttest (def mean1, def mean2, def sd1, def sd2, def n, def m) {
     return (mean1 - mean2) / Math.sqrt((sd1 * sd1) / n + (sd2 * sd2) / m)
}

def probt(def degree, def t) {
    return Probability.studentT(degree, t)
}


def nmap = [:] // ENSG to Name
def pcount = 0
def map = [:].withDefault { [:].withDefault { new DoubleArrayList() } } //gene -> cancer -> rank[]
new File("complete_ranked_genes_all_samples/").eachFile { file ->
//    println "$pcount patients processed."
    pcount += 1
    def first = true
    def count = 0
    def patient = file.toString().replaceAll(".txt", "").replaceAll("complete_ranked_genes_all_samples/","")
    file.splitEachLine("\t") { line ->
	if (first) {
	    first = false
	} else {
	    def gene = line[0]
	    def cancer = line[1]
	    def gname = line[2]
	    if (cancer.startsWith("TCGA")) {
		map[gene][cancer].add(count)
		nmap[gene] = gname
		count += 1
	    }
	}
    }
}

map.each { gene, map2 ->
    def l2 = new DoubleArrayList()
    map2.each { cancer, l ->
	def mean = Descriptive.mean(l)
	def sd = Descriptive.standardDeviation(Descriptive.variance(l.size(), Descriptive.sum(l), Descriptive.sumOfSquares(l)))
	def m = l.size()

	// test against uniform random
	def mean2 = 0.5 * m // (1/2) * (a + b)
	def sd2 = Descriptive.standardDeviation((1/12) * (m-1) * (m-1)) // (1/12) * (b-a)^2

	if (m > 2) {
	    try {
		def pval = probt(m-1.0, ttest(mean, mean2, sd, sd2, m, m))
		if ((pval * map.keySet().size() * 33) < 0.05 ) { // Bonferroni correction
		    println "GLOBAL\t$gene\t${nmap[gene]}\t$cancer\t"+pval+"\t"+(pval * map.keySet().size() * 33)
		}
	    } catch (Exception E) {
		E.printStackTrace()
	    }
	}
	l2.addAllOf(l)
    }
    // test against ranks in other cancers (compare against l2)
    def mean2 = Descriptive.mean(l2)
    def sd2 = Descriptive.standardDeviation(Descriptive.variance(l2.size(), Descriptive.sum(l2), Descriptive.sumOfSquares(l2)))
    def m2 = l2.size()
    map2.each { cancer, l ->
	def mean = Descriptive.mean(l)
	def sd = Descriptive.standardDeviation(Descriptive.variance(l.size(), Descriptive.sum(l), Descriptive.sumOfSquares(l)))
	def m = l.size()
	if (m > 2) {
	    try {
		def pval = probt(m-1.0, ttest(mean, mean2, sd, sd2, m, m2))
		if ((pval * map.keySet().size() * 33) < 0.05 ) { // Bonferroni correction
		    println "SPECIFIC\t$gene\t${nmap[gene]}\t$cancer\t"+pval+"\t"+pval*map.keySet().size() * 33
		}
	    } catch (Exception E) {
		E.printStackTrace()
		exit()
	    }
	}
    }
}

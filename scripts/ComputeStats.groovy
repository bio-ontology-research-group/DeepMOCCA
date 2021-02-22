@Grab(group='colt', module='colt', version='1.2.0')
@Grab(group='org.apache.commons', module='commons-math3', version='3.5')
import cern.colt.list.*
import cern.jet.stat.Descriptive
import cern.jet.stat.Probability
import org.apache.commons.math3.stat.inference.*

def genes = new LinkedHashSet()

TTest tTest = new TTest()

def nmap = [:] // ENSG to Name
def map = [:].withDefault { [:].withDefault { [] } } //gene -> cancer -> rank[]
new File("complete_ranked_genes_all_samples-new/").eachFile { file ->
    def first = true
    def count = 0
    def cancer = ""
    file.splitEachLine("\t") { line ->
	if (first) {
	    first = false
	} else {
	    def gene = line[0]
	    cancer = line[1]
	    def gname = line[2]
	    map[gene][cancer] << count
	    nmap[gene] = gname
	    count += 1
	}
    }
}

// get cohort sizes
def cmap = [:].withDefault { 0 }
map.each { gene, map2 ->
    map2.each { cancer, l ->
	if (cmap[cancer] < l.size()) {
	    cmap[cancer] = l.size()
	}
    }
}

def gsize = nmap.keySet().size()

map.each { gene, map2 ->
    map2.each { cancer, l ->
	Double mean2 = ( 17186 ) / 2

	// add the missing values (omitted through pooling) -> they are ranked bottom
	(cmap[cancer] - l.size()).times {
	    l << 17186 * 1 // 17186
	}
	
	def mean = l.sum() / l.size()

	if (l.size() > 2) {
	    def pval = tTest.tTest(mean2, (Double[])l.toArray()) * 17186 * 33  // with Bonferroni correction; two-sided
	    if ((pval ) < 0.05 && mean < mean2 ) { 
		println "$gene\t${nmap[gene]}\t$cancer\t"+pval + "\t" + mean
	    }
	}
    }
}

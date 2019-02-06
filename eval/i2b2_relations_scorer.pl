#!/usr/bin/perl -w
#
#
#  Author: Geeticka Chauhan
#          MIT CSAIL
#
#  WHAT: This is a scorer for DDI 2013 Extraction task 9.2 modified from the scorer of Semeval 2010 Task #8 by
#  Preslav Nakov at National University of Singapore
#
#
#  Last modified: February 5, 2018
#
#  Current version: 1.1
#
#
#
#  Use:
#     i2b2_relations_scorer.pl <PROPOSED_ANSWERS> <ANSWER_KEY>
#
#  Example2:
#     i2b2_relations_scorer.pl proposed_answer1.txt answer_key1.txt > result_scores1.txt
#     i2b2_relations_scorer.pl proposed_answer2.txt answer_key2.txt > result_scores2.txt
#     i2b2_relations_scorer.pl proposed_answer3.txt answer_key3.txt > result_scores3.txt
#
#  Description:
#     The scorer takes as input a proposed classification file and an answer key file.
#     Both files should contain one prediction per line in the format "<SENT_ID>	<RELATION>"
#     with a TAB as a separator, e.g.,
#           1	TrIP
#           2	TrWP
#           3	TrCP
#               ...
#     The files do not have to be sorted in any way and the first file can have predictions
#     for a subset of the IDs in the second file only, e.g., because hard examples have been skipped.
#     Repetitions of IDs are not allowed in either of the files.
#
#     The scorer calculates and outputs the following statistics:
#        (1) confusion matrix, which shows
#           - the sums for each row/column: -SUM-
#           - the number of skipped examples: skip
#           - the number of examples with correct relation, but wrong directionality: xDIRx
#           - the number of examples in the answer key file: ACTUAL ( = -SUM- + skip + xDIRx )
#        (2) accuracy and coverage
#        (3) precision (P), recall (R), and F1-score for each relation
#        (4) micro-averaged P, R, F1, where the calculations ignore the Other category.
#        (5) macro-averaged P, R, F1, where the calculations ignore the Other category.
#
#     Note that in scores (4) and (5), skipped examples are equivalent to those classified as Other.
#     So are examples classified as relations that do not exist in the key file (which is probably not optimal).
#
#     The scoring is done three times:
#       (i)   as a (2*9+1)-way classification
#       (ii)  as a (9+1)-way classification, with directionality ignored
#       (iii) as a (9+1)-way classification, with directionality taken into account.
#     
#     The official score is the micro-averaged F1-score for (iii).
#

use strict;


###############
###   I/O   ###
###############

if ($#ARGV != 1) {
	die "Usage:\ni2b2_relations_scorer.pl <PROPOSED_ANSWERS> <ANSWER_KEY>\n";
}

my $PROPOSED_ANSWERS_FILE_NAME = $ARGV[0];
my $ANSWER_KEYS_FILE_NAME      = $ARGV[1];


################
###   MAIN   ###
################

my (%confMatrix8way, %confMatrix3way) = ();
my (%idsProposed, %idsAnswer) = ();
my (%allLabels8wayAnswer, %allLabels3wayAnswer) = ();
my (%allLabels8wayProposed, %allLabels3wayProposed) = ();

### 1. Read the file contents
my $totalProposed = &readFileIntoHash($PROPOSED_ANSWERS_FILE_NAME, \%idsProposed);
my $totalAnswer = &readFileIntoHash($ANSWER_KEYS_FILE_NAME, \%idsAnswer);

### 2. Calculate the confusion matrices
foreach my $id (keys %idsProposed) {

	### 2.1. Unexpected IDs are not allowed
	die "File $PROPOSED_ANSWERS_FILE_NAME contains a bad ID: '$id'"
		if (!defined($idsAnswer{$id}));

	### 2.2. Update the 8-way confusion matrix
	my $labelProposed = $idsProposed{$id};
	my $labelAnswer   = $idsAnswer{$id};
	$confMatrix8way{$labelProposed}{$labelAnswer}++;
	$allLabels8wayProposed{$labelProposed}++;

        ### 2.3 Update the 3-way confusion matrix
	my $labelProposed3way = $idsProposed{$id};
	my $labelAnswer3way   = $idsAnswer{$id};
        $labelProposed3way = &getThreeWayEval($labelProposed3way);
        $labelAnswer3way = &getThreeWayEval($labelAnswer3way);
	$confMatrix3way{$labelProposed3way}{$labelAnswer3way}++;
	$allLabels3wayProposed{$labelProposed3way}++;

}

### 3. Calculate the ground truth distributions
foreach my $id (keys %idsAnswer) {

	### 3.1. Update the 8-way answer distribution
	my $labelAnswer = $idsAnswer{$id};
	$allLabels8wayAnswer{$labelAnswer}++;

	### 3.2. Update the 3-way answer distribution
	my $labelAnswer3way = $labelAnswer;
        $labelAnswer3way = &getThreeWayEval($labelAnswer3way);
	$allLabels3wayAnswer{$labelAnswer3way}++;
}

### 4. Check for proposed classes that are not contained in the answer key file: this may happen in cross-validation
foreach my $labelProposed (sort keys %allLabels8wayProposed) {
	if (!defined($allLabels8wayAnswer{$labelProposed})) {
		print "!!!WARNING!!! The proposed file contains $allLabels8wayProposed{$labelProposed} label(s) of type '$labelProposed', which is NOT present in the key file.\n\n";
	}
}

### 4. 8-way evaluation
print "<<< 8-WAY EVALUATION >>>:\n\n";
&printConfusionMatrix(\%confMatrix8way, \%allLabels8wayProposed, \%allLabels8wayAnswer, $totalProposed, $totalAnswer);

my ($microF1, $microF1_ProbTreat, $microF1_ProbTest, $microF1_ProbProb) = &evaluate(\%confMatrix8way, \%allLabels8wayProposed, \%allLabels8wayAnswer, $totalProposed, $totalAnswer);



### 7. Output the 4 micro F1 values
printf "<<< The 8-way evaluation: micro-averaged F1 = %0.2f%s >>>\n", $microF1, '%';
printf "<<< The Problem-Treatment evaluation: micro-averaged F1 = %0.2f%s >>>\n", $microF1_ProbTreat, '%';
printf "<<< The Problem-Test evaluation: micro-averaged F1 = %0.2f%s >>>\n", $microF1_ProbTest, '%';
printf "<<< The Problem-Problem evaluation: micro-averaged F1 = %0.2f%s >>>\n", $microF1_ProbProb, '%';


################
###   SUBS   ###
################

sub getIDandLabel() {
	my $line = shift;
	return (-1,()) if ($line !~ /^([0-9]+)\t([^\r]+)\r?\n$/);

	my ($id, $label) = ($1, $2);


	return ($id, $label)
    if (($label eq 'TrIP') || ($label eq 'TrWP') || ($label eq 'TrCP') ||
		($label eq 'TrAP') || ($label eq 'TrNAP') || ($label eq 'TeRP') ||
		($label eq 'TeCP')   || ($label eq 'PIP'));
	
	return (-1, ());
}

sub getThreeWayEval() {
      my ($label) = @_;
      return 'Prob-Treat'
    if (($label eq 'TrIP') || ($label eq 'TrWP') || ($label eq 'TrCP') ||
		($label eq 'TrAP')   || ($label eq 'TrNAP'));

      return 'Prob-Test'
    if (($label eq 'TeRP') || ($label eq 'TeCP'));

      return 'Prob-Prob' if ($label eq 'PIP');
      return ();
}


sub readFileIntoHash() {
	my ($fname, $ids) = @_;
	open(INPUT, $fname) or die "Failed to open $fname for text reading.\n";
	my $lineNo = 0;
	while (<INPUT>) {
		$lineNo++;
		my ($id, $label) = &getIDandLabel($_);
		die "Bad file format on line $lineNo: '$_'\n" if ($id < 0);
		if (defined $$ids{$id}) {
			s/[\n\r]*$//;
			die "Bad file format on line $lineNo (ID $id is already defined): '$_'\n";
		}
		$$ids{$id} = $label;
	}
	close(INPUT) or die "Failed to close $fname.\n";
	return $lineNo;
}


sub printConfusionMatrix() {
	my ($confMatrix, $allLabelsProposed, $allLabelsAnswer, $totalProposed, $totalAnswer) = @_;

	### 0. Create a merged list for the confusion matrix
	my @allLabels = ();
	&mergeLabelLists($allLabelsAnswer, $allLabelsProposed, \@allLabels);

	### 1. Print the confusion matrix heading
	print "Confusion matrix:\n";
	print "       ";
	foreach my $label (@allLabels) {
		printf " %5s", $label;
	}
	print " <-- classified as\n";
	print "      +";
	foreach my $label (@allLabels) {
		print "------";
	}
	
	print "+ -SUM- skip ACTUAL\n";
	

	### 2. Print the rest of the confusion matrix
	my $freqCorrect = 0;
	my $ind = 1;
	foreach my $labelAnswer (sort keys %{$allLabelsAnswer}) {

		### 2.1. Output the short relation label
		printf " %5s |", $labelAnswer;

		### 2.2. Output a row of the confusion matrix
		my $sumProposed = 0;
		foreach my $labelProposed (@allLabels) {
			$$confMatrix{$labelProposed}{$labelAnswer} = 0
				if (!defined($$confMatrix{$labelProposed}{$labelAnswer}));
			printf "%5d ", $$confMatrix{$labelProposed}{$labelAnswer};
			$sumProposed += $$confMatrix{$labelProposed}{$labelAnswer};
		}

		### 2.3. Output the horizontal sums
		my $ans = defined($$allLabelsAnswer{$labelAnswer}) ? $$allLabelsAnswer{$labelAnswer} : 0;
		printf "| %5d %5d %5d\n", $sumProposed, $ans - $sumProposed, $ans;

		$ind++;

		$$confMatrix{$labelAnswer}{$labelAnswer} = 0
			if (!defined($$confMatrix{$labelAnswer}{$labelAnswer}));
		$freqCorrect += $$confMatrix{$labelAnswer}{$labelAnswer};
	}
	print "      +";
	foreach (@allLabels) {
		print "------";
	}
	print "+\n";
	
	### 3. Print the vertical sums
	print " -SUM-  ";
	foreach my $labelProposed (@allLabels) {
		$$allLabelsProposed{$labelProposed} = 0
			if (!defined $$allLabelsProposed{$labelProposed});
		printf "%5d ", $$allLabelsProposed{$labelProposed};
	}
	printf "  %5d %5d %5d\n\n", $totalProposed, $totalAnswer - $totalProposed, $totalAnswer;

	### 4. Output the coverage
	my $coverage = 100.0 * $totalProposed / $totalAnswer;
	printf "%s%d%s%d%s%5.2f%s", 'Coverage = ', $totalProposed, '/', $totalAnswer, ' = ', $coverage, "\%\n";

	### 5. Output the accuracy
	my $accuracy = 100.0 * $freqCorrect / $totalProposed;
	printf "%s%d%s%d%s%5.2f%s", 'Accuracy (calculated for the above confusion matrix) = ', $freqCorrect, '/', $totalProposed, ' = ', $accuracy, "\%\n";

	### 6. Output the accuracy considering all skipped to be wrong
	$accuracy = 100.0 * $freqCorrect / $totalAnswer;
	printf "%s%d%s%d%s%5.2f%s", 'Accuracy (considering all skipped examples as Wrong) = ', $freqCorrect, '/', $totalAnswer, ' = ', $accuracy, "\%\n";
}

sub evaluate(){
	my ($confMatrix, $allLabelsProposed, $allLabelsAnswer, $totalProposed, $totalAnswer) = @_;
	### 8. Output P, R, F1 for each relation
	my ($macroP, $macroR, $macroF1) = (0, 0, 0);
	my ($microCorrect, $microProposed, $microAnswer) = (0, 0, 0);
	
	my ($macroP_ProbTreat, $macroR_ProbTreat, $macroF1_ProbTreat) = (0, 0, 0);
	my ($microCorrectProbTreat, $microProposedProbTreat, $microAnswerProbTreat) = (0, 0, 0);

	my ($macroP_ProbTest, $macroR_ProbTest, $macroF1_ProbTest) = (0, 0, 0);
	my ($microCorrectProbTest, $microProposedProbTest, $microAnswerProbTest) = (0, 0, 0);
	
	my ($macroP_ProbProb, $macroR_ProbProb, $macroF1_ProbProb) = (0, 0, 0);
	my ($microCorrectProbProb, $microProposedProbProb, $microAnswerProbProb) = (0, 0, 0);
	print "\nResults for the individual relations:\n";
	foreach my $labelAnswer (sort keys %{$allLabelsAnswer}) {


		### 8.1. Prevent Perl complains about unintialized values
		if (!defined($$allLabelsProposed{$labelAnswer})) {
			$$allLabelsProposed{$labelAnswer} = 0;
		}

		### 8.1. Calculate P/R/F1
		my $P  = (0 == $$allLabelsProposed{$labelAnswer}) ? 0
				: 100.0 * $$confMatrix{$labelAnswer}{$labelAnswer} / $$allLabelsProposed{$labelAnswer};
		my $R  = (0 == $$allLabelsAnswer{$labelAnswer}) ? 0
				: 100.0 * $$confMatrix{$labelAnswer}{$labelAnswer} / $$allLabelsAnswer{$labelAnswer};
		my $F1 = (0 == $P + $R) ? 0 : 2 * $P * $R / ($P + $R);

		### 8.3. Output P/R/F1
		printf "%25s%s%4d%s%4d%s%6.2f", $labelAnswer,
			" :    P = ", $$confMatrix{$labelAnswer}{$labelAnswer}, '/', $$allLabelsProposed{$labelAnswer}, ' = ', $P;
		printf"%s%4d%s%4d%s%6.2f%s%6.2f%s\n",
		  	 "%     R = ", $$confMatrix{$labelAnswer}{$labelAnswer}, '/', $$allLabelsAnswer{$labelAnswer},   ' = ', $R,
			 "%     F1 = ", $F1, '%';

		### 8.5. Accumulate statistics for micro/macro-averaging
		$macroP  += $P;
		$macroR  += $R;
		$macroF1 += $F1;
		$microCorrect += $$confMatrix{$labelAnswer}{$labelAnswer};
		$microProposed += $$allLabelsProposed{$labelAnswer};
		$microAnswer += $$allLabelsAnswer{$labelAnswer};
		if (($labelAnswer eq 'TrIP') || ($labelAnswer eq 'TrWP') || ($labelAnswer eq 'TrCP') ||
			($labelAnswer eq 'TrAP') || ($labelAnswer eq 'TrNAP')){
			$macroP_ProbTreat += $P;
			$macroR_ProbTreat += $R;
			$macroF1_ProbTreat += $F1;
			$microCorrectProbTreat += $$confMatrix{$labelAnswer}{$labelAnswer};
			$microProposedProbTreat += $$allLabelsProposed{$labelAnswer};
			$microAnswerProbTreat += $$allLabelsAnswer{$labelAnswer};
		}
		elsif (($labelAnswer eq 'TeRP') || ($labelAnswer eq 'TeCP')){
			$macroP_ProbTest += $P;
			$macroR_ProbTest += $R;
			$macroF1_ProbTest += $F1;
			$microCorrectProbTest += $$confMatrix{$labelAnswer}{$labelAnswer};
			$microProposedProbTest += $$allLabelsProposed{$labelAnswer};
			$microAnswerProbTest += $$allLabelsAnswer{$labelAnswer};
		}
		elsif ($labelAnswer eq 'PIP'){
			$macroP_ProbProb += $P;
			$macroR_ProbProb += $R;
			$macroF1_ProbProb += $F1;
			$microCorrectProbProb += $$confMatrix{$labelAnswer}{$labelAnswer};
			$microProposedProbProb += $$allLabelsProposed{$labelAnswer};
			$microAnswerProbProb += $$allLabelsAnswer{$labelAnswer};
		}
	}
	
	### 9. Output the micro-averaged and macro averaged P, R, F1 for 8 way eval
	my $microP  = (0 == $microProposed)    ? 0 : 100.0 * $microCorrect / $microProposed;
	my $microR  = (0 == $microAnswer)      ? 0 : 100.0 * $microCorrect / $microAnswer;
	my $microF1 = (0 == $microP + $microR) ? 0 :   2.0 * $microP * $microR / ($microP + $microR);
        print "\nMicro-averaged result :\n";
	printf "%s%4d%s%4d%s%6.2f%s%4d%s%4d%s%6.2f%s%6.2f%s\n",
		      "P = ", $microCorrect, '/', $microProposed, ' = ', $microP,
		"%     R = ", $microCorrect, '/', $microAnswer, ' = ', $microR,
		"%     F1 = ", $microF1, '%';

	my $distinctLabelsCnt = keys %{$allLabelsAnswer}; 
	

	$macroP  /= $distinctLabelsCnt; # first divide by the number of categories
	$macroR  /= $distinctLabelsCnt;
	$macroF1 /= $distinctLabelsCnt;
	print "\nMACRO-averaged result :\n";
	printf "%s%6.2f%s%6.2f%s%6.2f%s\n\n\n\n", "P = ", $macroP, "%\tR = ", $macroR, "%\tF1 = ", $macroF1, '%';
	
	### 10. Output the micro-averaged and macro averaged P, R, F1 for Prob-Treat
	print "<<< Problem-Treatment Relations >>>:\n\n";
	
	my $microP_ProbTreat  = (0 == $microProposedProbTreat)    ? 0 : 100.0 * $microCorrectProbTreat / $microProposedProbTreat;
	my $microR_ProbTreat  = (0 == $microAnswerProbTreat)      ? 0 : 100.0 * $microCorrectProbTreat / $microAnswerProbTreat;
	my $microF1_ProbTreat = (0 == $microP_ProbTreat + $microR_ProbTreat) ? 0 :   2.0 * $microP_ProbTreat * $microR_ProbTreat / ($microP_ProbTreat + $microR_ProbTreat);
        print "\nMicro-averaged result :\n";
	printf "%s%4d%s%4d%s%6.2f%s%4d%s%4d%s%6.2f%s%6.2f%s\n",
		      "P = ", $microCorrectProbTreat, '/', $microProposedProbTreat, ' = ', $microP_ProbTreat,
		"%     R = ", $microCorrectProbTreat, '/', $microAnswerProbTreat, ' = ', $microR_ProbTreat,
		"%     F1 = ", $microF1_ProbTreat, '%';

	$macroP_ProbTreat  /= $distinctLabelsCnt; # first divide by the number of categories
	$macroR_ProbTreat  /= $distinctLabelsCnt;
	$macroF1_ProbTreat /= $distinctLabelsCnt;
	print "\nMACRO-averaged result :\n";
	printf "%s%6.2f%s%6.2f%s%6.2f%s\n\n\n\n", "P = ", $macroP_ProbTreat, "%\tR = ", $macroR_ProbTreat, "%\tF1 = ", $macroF1_ProbTreat, '%';
	

	### 11. Output the micro-averaged and macro averaged P, R, F1 for Prob-Test
	print "<<< Problem-Test Relations >>>:\n\n";
	
	my $microP_ProbTest  = (0 == $microProposedProbTest)    ? 0 : 100.0 * $microCorrectProbTest / $microProposedProbTest;
	my $microR_ProbTest  = (0 == $microAnswerProbTest)      ? 0 : 100.0 * $microCorrectProbTest / $microAnswerProbTest;
	my $microF1_ProbTest = (0 == $microP_ProbTest + $microR_ProbTest) ? 0 :   2.0 * $microP_ProbTest * $microR_ProbTest / ($microP_ProbTest + $microR_ProbTest);
        print "\nMicro-averaged result :\n";
	printf "%s%4d%s%4d%s%6.2f%s%4d%s%4d%s%6.2f%s%6.2f%s\n",
		      "P = ", $microCorrectProbTest, '/', $microProposedProbTest, ' = ', $microP_ProbTest,
		"%     R = ", $microCorrectProbTest, '/', $microAnswerProbTest, ' = ', $microR_ProbTest,
		"%     F1 = ", $microF1_ProbTest, '%';

	$macroP_ProbTest  /= $distinctLabelsCnt; # first divide by the number of categories
	$macroR_ProbTest  /= $distinctLabelsCnt;
	$macroF1_ProbTest /= $distinctLabelsCnt;
	print "\nMACRO-averaged result :\n";
	printf "%s%6.2f%s%6.2f%s%6.2f%s\n\n\n\n", "P = ", $macroP_ProbTest, "%\tR = ", $macroR_ProbTest, "%\tF1 = ", $macroF1_ProbTest, '%';
	
	### 12. Output the micro-averaged and macro averaged P, R, F1 for Prob-Prob
	print "<<< Problem-Problem Relations >>>:\n\n";
	
	my $microP_ProbProb  = (0 == $microProposedProbProb)    ? 0 : 100.0 * $microCorrectProbProb / $microProposedProbProb;
	my $microR_ProbProb  = (0 == $microAnswerProbProb)      ? 0 : 100.0 * $microCorrectProbProb / $microAnswerProbProb;
	my $microF1_ProbProb = (0 == $microP_ProbProb + $microR_ProbProb) ? 0 :   2.0 * $microP_ProbProb * $microR_ProbProb / ($microP_ProbProb + $microR_ProbProb);
        print "\nMicro-averaged result :\n";
	printf "%s%4d%s%4d%s%6.2f%s%4d%s%4d%s%6.2f%s%6.2f%s\n",
		      "P = ", $microCorrectProbProb, '/', $microProposedProbProb, ' = ', $microP_ProbProb,
		"%     R = ", $microCorrectProbProb, '/', $microAnswerProbProb, ' = ', $microR_ProbProb,
		"%     F1 = ", $microF1_ProbProb, '%';

	$macroP_ProbProb  /= $distinctLabelsCnt; # first divide by the number of categories
	$macroR_ProbProb  /= $distinctLabelsCnt;
	$macroF1_ProbProb /= $distinctLabelsCnt;
	print "\nMACRO-averaged result :\n";
	printf "%s%6.2f%s%6.2f%s%6.2f%s\n\n\n\n", "P = ", $macroP_ProbProb, "%\tR = ", $macroR_ProbProb, "%\tF1 = ", $macroF1_ProbProb, '%';
	
	return ($microF1, $microF1_ProbTreat, $microF1_ProbTest, $microF1_ProbProb);
}



sub mergeLabelLists() {
	my ($hash1, $hash2, $mergedList) = @_;
	foreach my $key (sort keys %{$hash1}) {
		push @{$mergedList}, $key if ($key ne 'WRONG_DIR');
	}
	foreach my $key (sort keys %{$hash2}) {
		push @{$mergedList}, $key if (($key ne 'WRONG_DIR') && !defined($$hash1{$key}));
	}
}

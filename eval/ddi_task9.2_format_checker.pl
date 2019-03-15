#!/usr/bin/perl -w
#
#
#  Author: Geeticka Chauhan
#          MIT CSAIL
#          geeticka@mit.edu
#
#  WHAT: This is an output file checker for the DDI Extraction 2013 task, modified from that of the
#  SemEval-2010 Task #8 by Preslav Nakov at National University of Singapore.
#
#  Use:
#     ddi_task9.2_format_checker.pl <PROPOSED_ANSWERS>
#
#  Examples:
#     ddi_task9.2_format_checker.pl proposed_answer1.txt
#     ddi_task9.2_format_checker.pl proposed_answer2.txt
#     ddi_task9.2_format_checker.pl proposed_answer3.txt
#     ddi_task9.2_format_checker.pl proposed_answer4.txt
#
#   In the examples above, the first three files are OK, while the last one contains four errors.
#   And answer_key2.txt contains the true labels for the *training* dataset.
#
#  Description:
#     The scorer takes as input a proposed classification file,
#     which should contain one prediction per line in the format "<SENT_ID>	<RELATION>"
#     with a TAB as a separator, e.g.,
#           1	int
#           2	mechanism
#           3	none
#               ...
#     The file does not have to be sorted in any way.
#     Repetitions of IDs are not allowed.
#
#     In case of problems, the checker outputs the problemtic line and its number.
#     Finally, the total number of problems found is reported
#     or a message is output saying that the file format is OK.
#
#     Participants are expected to check their output using this checker before submission.
#
#  Last modified: Dec 28, 2018
#
#

use strict;

###############
###   I/O   ###
###############

if ($#ARGV != 0) {
	die "Usage:\nddi_task9.2_format_checker.pl <PROPOSED_ANSWERS>\n";
}

my $INPUT_FILE_NAME = $ARGV[0];

################
###   MAIN   ###
################
my %ids = ();

my $errCnt = 0;
open(INPUT, $INPUT_FILE_NAME) or die "Failed to open $INPUT_FILE_NAME for text reading.\n";
for (my $lineNo = 1; <INPUT>; $lineNo++) {
	my ($id, $label) = &getIDandLabel($_);
	if ($id < 0) {
		s/[\n\r]*$//;
		print "Bad file format on line $lineNo: '$_'\n";
		$errCnt++;
	}
	elsif (defined $ids{$id}) {
		s/[\n\r]*$//;
		print "Bad file format on line $lineNo (ID $id is already defined): '$_'\n";
		$errCnt++;
	}
	$ids{$id}++;
}
close(INPUT) or die "Failed to close $INPUT_FILE_NAME.\n";

if (0 == $errCnt) {
	print "\n<<< The file format is OK.\n";
}
else {
	print "\n<<< The format is INCORRECT: $errCnt problematic line(s) found!\n";
}


################
###   SUBS   ###
################

sub getIDandLabel() {
	my $line = shift;

	return (-1,()) if ($line !~ /^([0-9]+)\t([^\r]+)\r?\n$/);
	my ($id, $label) = ($1, $2);

    return ($id, '_none') if ($label eq 'none');

	return ($id, $label)
    if (($label eq 'advise')       || ($label eq 'effect')       ||
		($label eq 'mechanism')   || ($label eq 'int'));
	
	return (-1, ());
}

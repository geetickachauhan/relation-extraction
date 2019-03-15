#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk

syntactic_compiled_grammar = {}


class PatternGrammar:
    @property
    def syntactic_grammars(self):
        grammar = {
            0: """
                JJ_VBG_RB_DESCRIBING_NN: {   (<CC|,>?<JJ|JJ.>*<VB.|V.>?<NN|NN.>)+<RB|RB.>*<MD>?<WDT|DT>?<VB|VB.>?<RB|RB.>*(<CC|,>?<RB|RB.>?<VB|VB.|JJ.|JJ|RB|RB.>+)+}
                """,
            1: """
                    VBG_DESRIBING_NN: {<NN|NN.><VB|VB.>+<RB|RB.>*<VB|VB.>}
                """,
        }
        return grammar

    def get_syntactic_grammar(self, index):
        global syntactic_compiled_grammar
        compiled_grammar = syntactic_compiled_grammar.get(index, None)
        if compiled_grammar is None:
            compiled_grammar = self.compile_syntactic_grammar(index)
            syntactic_compiled_grammar[index] = compiled_grammar
        return compiled_grammar

    def compile_syntactic_grammar(self, index):
        return nltk.RegexpParser(self.syntactic_grammars[index])

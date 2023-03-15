# mcts debugging helpful hint
[node.name, node.Qreward, node.Nvisit, node.player_to_move, [x.prior_prob for x in node.children], [y.game_over for y in node.children]]
[self.tree.root.name, self.tree.root.Qreward, self.tree.root.Nvisit, self.tree.root.player_to_move, [x.prior_prob for x in self.tree.root.children], [y.game_over for y in self.tree.root.children]]

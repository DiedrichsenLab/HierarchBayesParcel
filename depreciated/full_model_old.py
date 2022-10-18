class  FullMultiModel:
    def pre_train(self, Y, iter=10, emi_idx=None, prior=None, fit_arrangement=False):
        """Correcting the init parameters for all emission models by sampling from
           a prior or learnt from one of the emission models
        Args:
            Y: data
            iter: the number of iterations for fine-tuning the params
            emi_idx: the index of which emission model params should be used
            prior: if not None, correcting the inits from prior. shape (K, P)
            fit_arrangement: if True, fit arrangement model
        Returns:

        """
        # Initialize data to all emission models
        for n, e in enumerate(self.emissions):
            e.initialize(Y[n])

        # learn prior from the dataset with the highest dimensions?
        # or with the most subjects?
        if emi_idx is None:
            # dims = [e.V.shape[0] for e in self.emissions]  # dims
            dims = [e.num_subj for e in self.emissions]  # num_subj
            emi_idx = dims.index(max(dims))

        ground_em = self.emissions[emi_idx]
        if prior is None:
            for i in range(iter):
                # Get the (approximate) posterior p(U|Y)
                emloglik = ground_em.Estep()
                Uhat, _ = self.arrange.Estep(emloglik)
                ground_em.Mstep(Uhat)

                if fit_arrangement:
                    self.arrange.Mstep()

            Uhat = pt.mean(Uhat, dim=0)
            for em in self.emissions:
                if em is not ground_em:
                    em.Mstep(Uhat.unsqueeze(0).repeat(em.num_subj, 1, 1))
        else:
            for em in self.emissions:
                em.Mstep(prior.unsqueeze(0).repeat(em.num_subj, 1, 1))


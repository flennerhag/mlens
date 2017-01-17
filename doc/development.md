# Development

## Front end roadmap

The project is rapidly progressing. The parallelized backend is in place so the coming taks is to develop the front-end API for different types of ensembles need to be built. This however is a relatively straightforward task so expect major additions soon. In the pipeline of Ensembles to be implemented are currently: 

- Blending
- Super Learner
- Subsemble

## Back end roadmap

Currently, **parallelization** is maximized by pre-making the preprocessed folds / data. This however will not scale well to very large datasets, and an alternative is to cache fitted preprocessing pipelines, to avoid repeatedly fitting them. This should not be a hard task to accomplish, as it only requires storing the fitted pipelines and calling them when necessary.

Finally, the current structure creates one hidden layer. Of course, by pipelining several ``Ensemble`` classes, a multi-layer ensemble can be created. This however must be done explicitly by the user, but implementing a ``add layer`` method to ensemble classes would make a more intuitive API.

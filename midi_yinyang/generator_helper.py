
def end_generator(generator):
    try:
        next(generator)
        raise ValueError('Generator did not end')
    except StopIteration as e:
        return e.value

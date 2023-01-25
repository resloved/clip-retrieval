"""The runner combine reader, mapper and writer to produce clip embeddings"""

import time


class Sampler:
    """Sampler"""

    def __init__(self, output_partition_id, output_partition_count):
        self.output_partition_id = output_partition_id
        self.output_partition_count = output_partition_count

    def __call__(self, l):
        return [
            e
            for i, e in enumerate(l)
            if i % self.output_partition_count == self.output_partition_id
        ]


class Runner:
    """Runner class"""

    def __init__(
        self,
        reader_builder,
        mapper_builder,
        writer_builder,
        logger_builder,
        output_partition_count,
    ):
        self.reader_builder = reader_builder
        self.mapper_builder = mapper_builder
        self.writer_builder = writer_builder
        self.logger_builder = logger_builder
        self.output_partition_count = output_partition_count

    def __call__(self, i):
        sampler = Sampler(i, self.output_partition_count)
        reader = self.reader_builder(sampler)
        writer = self.writer_builder(i)
        mapper = self.mapper_builder()
        logger = self.logger_builder(i)
        logger.start()
        iterator = reader.__iter__()
        while True:
            begin_time = time.time()
            start_time = time.perf_counter()
            try:
                batch = iterator.__next__()
            except StopIteration:
                break
            read_duration = time.perf_counter() - start_time
            start_time = time.perf_counter()
            embeddings = mapper(batch)
            inference_duration = time.perf_counter() - start_time
            start_time = time.perf_counter()
            writer(embeddings)
            write_duration = time.perf_counter() - start_time
            end_time = time.time()

            if "image_filename" in batch:
                sample_count = len(batch["image_filename"])
            elif "text_tokens" in batch:
                sample_count = batch["text_tokens"].shape[0]
            else:
                sample_count = len(batch["metadata"])

            logger(
                {
                    "start_time": begin_time,
                    "end_time": end_time,
                    "read_duration": read_duration,
                    "inference_duration": inference_duration,
                    "write_duration": write_duration,
                    "total_duration": end_time - begin_time,
                    "sample_count": sample_count,
                }
            )
        logger.end()
        writer.flush()

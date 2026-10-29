/* Temporary LD_PRELOAD timing probe. Logs no stream URLs or packet contents. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <libavformat/avformat.h>
#include <stdio.h>
#include <time.h>

static _Thread_local unsigned packets;
static _Thread_local int video_seen;
static _Thread_local int key_seen;

static double now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int avformat_open_input(AVFormatContext **ctx, const char *url,
                        const AVInputFormat *format, AVDictionary **options)
{
  int (*next)(AVFormatContext **, const char *, const AVInputFormat *, AVDictionary **)
      = dlsym(RTLD_NEXT, "avformat_open_input");
  double start = now_ms();
  fprintf(stderr, "ZAPTRACE open_begin mono_ms=%.3f\n", start);
  int result = next(ctx, url, format, options);
  fprintf(stderr, "ZAPTRACE open_end mono_ms=%.3f elapsed_ms=%.3f result=%d format=%s\n",
          now_ms(), now_ms() - start, result,
          result >= 0 && *ctx && (*ctx)->iformat ? (*ctx)->iformat->name : "none");
  packets = 0;
  video_seen = key_seen = 0;
  return result;
}

int avformat_find_stream_info(AVFormatContext *ctx, AVDictionary **options)
{
  int (*next)(AVFormatContext *, AVDictionary **) = dlsym(RTLD_NEXT, "avformat_find_stream_info");
  double start = now_ms();
  fprintf(stderr, "ZAPTRACE probe_begin mono_ms=%.3f\n", start);
  int result = next(ctx, options);
  fprintf(stderr, "ZAPTRACE probe_end mono_ms=%.3f elapsed_ms=%.3f result=%d\n",
          now_ms(), now_ms() - start, result);
  return result;
}

void avformat_close_input(AVFormatContext **ctx)
{
  void (*next)(AVFormatContext **) = dlsym(RTLD_NEXT, "avformat_close_input");
  fprintf(stderr, "ZAPTRACE close mono_ms=%.3f\n", now_ms());
  next(ctx);
}

int av_read_frame(AVFormatContext *ctx, AVPacket *packet)
{
  int (*next)(AVFormatContext *, AVPacket *) = dlsym(RTLD_NEXT, "av_read_frame");
  int result = next(ctx, packet);
  if (result < 0)
    return result;
  ++packets;
  int video = packet->stream_index >= 0 && (unsigned)packet->stream_index < ctx->nb_streams &&
              ctx->streams[packet->stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;
  int key = video && (packet->flags & AV_PKT_FLAG_KEY);
  if (packets == 1 || (video && !video_seen) || (key && !key_seen))
    fprintf(stderr, "ZAPTRACE packet mono_ms=%.3f count=%u video=%d key=%d\n",
            now_ms(), packets, video, !!key);
  video_seen |= video;
  key_seen |= !!key;
  return result;
}

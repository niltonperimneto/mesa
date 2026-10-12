#ifndef KK_PSO_WAIT_H
#define KK_PSO_WAIT_H

#include "c11/threads.h"
#include "util/os_time.h"
#include <vulkan/vulkan_core.h>

struct kk_pso_request {
   int64_t deadline_ns;
   bool lookup_only;
   bool handoff;
};

static inline struct kk_pso_request
kk_pso_request_create(unsigned timeout_ms)
{
   return (struct kk_pso_request){
      .deadline_ns = timeout_ms ? os_time_get_nano() +
                                 (int64_t)timeout_ms * 1000000 : 0,
   };
}

static inline VkResult
kk_pso_wait(cnd_t *ready, mtx_t *lock, const bool *compiling,
            const struct kk_pso_request *request)
{
   while (*compiling) {
      int ret;
      if (!request || !request->deadline_ns) {
         ret = cnd_wait(ready, lock);
      } else {
         int64_t remaining = request->deadline_ns - os_time_get_nano();
         if (remaining <= 0)
            return VK_TIMEOUT;
         struct timespec until;
         if (clock_gettime(CLOCK_REALTIME, &until))
            return VK_ERROR_UNKNOWN;
         int64_t ns = until.tv_nsec + remaining;
         until.tv_sec += ns / 1000000000;
         until.tv_nsec = ns % 1000000000;
         ret = cnd_timedwait(ready, lock, &until);
      }
      if (ret != thrd_success && ret != thrd_timedout)
         return VK_ERROR_UNKNOWN;
   }
   return VK_SUCCESS;
}

#endif

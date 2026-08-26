import { useEffect, useRef } from 'react'

export default function EzvizPlayer({ active, config, setPlayerState, setPlayerError, audio = false }) {
  const hostRef = useRef(null)

  useEffect(() => {
    if (!active || !config || !hostRef.current) return undefined

    let cancelled = false
    let player
    setPlayerState('loading')
    setPlayerError('')

    import('ezuikit-js')
      .then(({ EZUIKitPlayer }) => {
        if (cancelled || !hostRef.current) return
        const width = Math.max(320, Math.floor(hostRef.current.clientWidth))
        player = new EZUIKitPlayer({
          id: 'ezviz-player',
          accessToken: config.accessToken,
          url: config.url,
          width,
          height: Math.floor(width * 9 / 16),
          template: 'pcLive',
          audio,
          handleSuccess: () => {
            if (!cancelled) setPlayerState('playing')
          },
          handleError: error => {
            if (cancelled) return
            console.error('EZUIKit 播放失败', error)
            const encrypted = error?.type === 'handleRunTimeInfoError' && error?.data?.nErrorCode === 5
            setPlayerState('error')
            setPlayerError(encrypted
              ? '设备已启用视频加密，需要设备验证码。'
              : '萤石视频播放失败，请确认设备在线并刷新播放授权。')
          },
        })
      })
      .catch(error => {
        if (cancelled) return
        console.error('EZUIKit 初始化失败', error)
        setPlayerState('error')
        setPlayerError('播放器初始化失败，请检查浏览器兼容性和萤石播放权限。')
      })

    return () => {
      cancelled = true
      try {
        player?.stop?.()
        player?.destroy?.()
      } catch (_) { /* 播放器释放失败不影响后端监控 */ }
      setPlayerState('idle')
    }
  }, [active, config, setPlayerError, setPlayerState, audio])

  return <div ref={hostRef} className="ezviz-player-host"><div id="ezviz-player" /></div>
}
